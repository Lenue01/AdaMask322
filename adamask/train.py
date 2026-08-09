import dataclasses
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm

from adamask.data import get_dataloader
from adamask.sample import sample


def get_lr(global_step, warmup_steps, max_lr, min_lr, total_steps):
    if global_step < warmup_steps:
        return max_lr * global_step / warmup_steps
    progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def print_token_stats(diffusion, tokenizer, k=10):
    counts = diffusion.total
    mean_ce = diffusion.loss_sum / (counts + 1e-8)
    seen_mask = counts > 0
    seen_ce = mean_ce[seen_mask]
    seen_counts = counts[seen_mask]
    seen_ids = seen_mask.nonzero(as_tuple=True)[0]
    if seen_ids.numel() < k:
        print("  Not enough token data yet.")
        return
    # Lowest mean CE = easiest, highest mean CE = hardest.
    top_vals, top_idx = seen_ce.topk(k, largest=False)
    bot_vals, bot_idx = seen_ce.topk(k, largest=True)
    top_ids = seen_ids[top_idx].tolist()
    bot_ids = seen_ids[bot_idx].tolist()
    top_counts = seen_counts[top_idx].tolist()
    bot_counts = seen_counts[bot_idx].tolist()
    print(f"  Easiest tokens (lowest mean CE loss):")
    for tid, ce, cnt in zip(top_ids, top_vals.tolist(), top_counts):
        tok = tokenizer.convert_ids_to_tokens(tid)
        print(f"    {tok!r:20s}  ce={ce:.3f}  count={int(cnt)}")
    print(f"  Hardest tokens (highest mean CE loss):")
    for tid, ce, cnt in zip(bot_ids, bot_vals.tolist(), bot_counts):
        tok = tokenizer.convert_ids_to_tokens(tid)
        print(f"    {tok!r:20s}  ce={ce:.3f}  count={int(cnt)}")


def save_checkpoint(model, optimizer, scaler, diffusion, epoch, path):
    """Save full training state so a run can be resumed after a crash or disconnect."""
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "diffusion_total": diffusion.total,
        "diffusion_loss_sum": diffusion.loss_sum,
    }, path)


def load_checkpoint(path, model, optimizer, scaler, diffusion, device):
    """Restore model, optimizer, scaler, and token-difficulty state from a checkpoint.

    Returns the epoch to resume from (the one after the last completed epoch).
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    diffusion.total = ckpt["diffusion_total"].to(device)
    diffusion.loss_sum = ckpt["diffusion_loss_sum"].to(device)
    return ckpt["epoch"] + 1


def build_val_batches(config):
    """Cache a fixed set of validation batches once, up front.

    Reused unchanged every epoch so validation always sees the same examples.
    """
    val_config = dataclasses.replace(config, split=config.val_split)
    val_loader = get_dataloader(val_config)
    val_iter = iter(val_loader)
    batches = []
    for _ in range(config.val_batches):
        try:
            batch = next(val_iter)
        except StopIteration:
            break
        tokens = batch["input_ids"].to(config.device)
        pad_mask = batch["attention_mask"].to(config.device) == 0
        batches.append((tokens, pad_mask))
    return batches


def compute_val_loss(model, diffusion, val_batches, config):
    """Plain (unweighted, unsmoothed) cross-entropy on cached validation batches.

    Re-seeds the corruption generator every call so the exact same masked inputs
    are used across epochs -- val loss changes then reflect the model only, not
    which tokens happened to get masked this time.
    """
    if not val_batches:
        return None

    model.eval()
    generator = torch.Generator(device=config.device).manual_seed(config.val_seed)
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for tokens, pad_mask in val_batches:
            t = torch.randint(
                1, config.steps + 1, (tokens.size(0),), device=config.device, generator=generator
            )
            x_t, is_masked = diffusion.corrupt(tokens, t, generator=generator)
            logits = model(x_t, t, key_padding_mask=pad_mask)
            loss_mask = is_masked & ~pad_mask
            if loss_mask.any():
                total_loss += F.cross_entropy(logits[loss_mask], tokens[loss_mask]).item()
                count += 1
    model.train()
    return total_loss / count if count else None


def print_epoch_samples(model, diffusion, config, epoch, num_samples=2):
    """Decode a few samples so text quality is visible after every epoch, not just at the end."""
    print(f"\n--- Epoch {epoch} samples ---")
    tokens = sample(model, diffusion, config, num_samples=num_samples, temperature=0.9)
    for i, row in enumerate(tokens.tolist()):
        text = config.tokenizer.decode(row, skip_special_tokens=False)
        print(f"  sample {i}: {text}")
    model.train()


def train(model, diffusion, dataloader, config, resume_path=None):
    """Train the model with diffusion masking and partial reveal simulation."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0, weight_decay=config.weight_decay
    )
    use_amp = config.device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    data_iter = iter(dataloader)
    val_batches = build_val_batches(config)

    start_epoch = 0
    if resume_path:
        start_epoch = load_checkpoint(resume_path, model, optimizer, scaler, diffusion, config.device)
        print(f"Resumed from {resume_path}, continuing at epoch {start_epoch}")

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(range(config.steps_per_epoch), desc=f"Epoch {epoch}")
        for step in loop:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move the input batch to the selected device.
            tokens = batch["input_ids"].to(config.device)
            attn_msk = batch["attention_mask"].to(config.device)
            pad_mask = attn_msk == 0
            # pad_mask is True at padding positions; the model should ignore these.

            # Update learning rate based on the current global step.
            global_step = epoch * config.steps_per_epoch + step
            lr = get_lr(global_step, config.warmup_steps, config.lr, config.lr / 10, config.total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Zero gradients at the start of each accumulation cycle.
            if step % config.accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            t = torch.randint(1, config.steps + 1, (tokens.size(0),), device=config.device)
            # Train the loss mask directly off diffusion.corrupt's own is_masked, with
            # no extra reveal step: sample.py never partially reveals tokens within a
            # step either, so x_t's visible corruption at timestep t must match
            # mask_rate(t) exactly here too. A prior version revealed a fixed 50% of
            # masked positions back to ground truth regardless of t, which halved the
            # actual corruption the model trained on relative to what it sees during
            # real sampling at the same t -- a train/inference mismatch that was worst
            # exactly where it matters most, at high-t (heavily masked) steps.
            x_t, is_masked = diffusion.corrupt(tokens, t)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x_t, t, key_padding_mask=pad_mask)

                # Compute loss only on tokens that are masked and not padding.
                loss_mask = (is_masked & ~pad_mask).bool()
                if loss_mask.any():
                    target = tokens[loss_mask]
                    # Weight each masked token's loss by how hard it has historically
                    # been for the model, instead of masking hard tokens more often --
                    # this pushes gradient signal toward hard tokens without changing
                    # what the model sees as input.
                    difficulty = diffusion.get_difficulty(target)
                    weight = 1.0 + config.difficulty_loss_scale * difficulty
                    per_token_loss = F.cross_entropy(
                        logits[loss_mask],
                        target,
                        label_smoothing=0.05,
                        reduction="none",
                    )
                    loss = (per_token_loss * weight).mean() / config.accum_steps
                else:
                    loss = None

            if loss is not None:
                scaler.scale(loss).backward()

            # Apply gradients every accum_steps iterations. This must not be
            # skipped even when this particular step had no loss, otherwise
            # gradients accumulated earlier in the cycle get silently dropped
            # by the next cycle's zero_grad.
            if (step + 1) % config.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            # Update token difficulty stats using the positions the model actually
            # trained on this step.
            diffusion.update(logits.detach(), tokens, is_masked)

            if loss is not None:
                running_loss += loss.item() * config.accum_steps
            loop.set_postfix(loss=f"{running_loss / (step + 1):.4f}")

        # Save model and print statistics periodically.
        if epoch % config.save_every_epochs == 0 or epoch == config.num_epochs - 1:
            save_checkpoint(model, optimizer, scaler, diffusion, epoch, f"masked_diffusion_epoch_{epoch}.pt")
            print(f"\n--- Epoch {epoch} Token Difficulty ---")
            print_token_stats(diffusion, config.tokenizer)

        val_loss = compute_val_loss(model, diffusion, val_batches, config)
        if val_loss is not None:
            print(f"  Val loss (fixed mask seed): {val_loss:.4f}")

        print_epoch_samples(model, diffusion, config, epoch)
        print()
