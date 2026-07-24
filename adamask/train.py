import math
import torch
import torch.nn.functional as F
from tqdm import tqdm


def get_lr(global_step, warmup_steps, max_lr, min_lr, total_steps):
    if global_step < warmup_steps:
        return max_lr * global_step / warmup_steps
    progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def apply_partial_reveal(x_t, tokens, is_masked, reveal_prob=0.5):
    """Reveal a random subset of masked positions from the original tokens.

    This simulates a partially decoded state during diffusion training.
    """
    if not is_masked.any():
        return x_t, is_masked

    reveal_mask = (torch.rand(tokens.shape, device=tokens.device) < reveal_prob) & is_masked
    x_t[reveal_mask] = tokens[reveal_mask]
    return x_t, is_masked & ~reveal_mask


def print_token_stats(diffusion, tokenizer, k=10):
    accuracy = diffusion.correct / (diffusion.total + 1e-8)
    seen_mask = diffusion.total > 1.0
    seen_acc = accuracy[seen_mask]
    seen_ids = seen_mask.nonzero(as_tuple=True)[0]
    if seen_ids.numel() < k:
        print("  Not enough token data yet.")
        return
    top_vals, top_idx = seen_acc.topk(k)
    bot_vals, bot_idx = seen_acc.topk(k, largest=False)
    top_ids = seen_ids[top_idx].tolist()
    bot_ids = seen_ids[bot_idx].tolist()
    print(f"  Easiest tokens (highest accuracy):")
    for tid, acc in zip(top_ids, top_vals.tolist()):
        tok = tokenizer.convert_ids_to_tokens(tid)
        print(f"    {tok!r:20s}  acc={acc:.3f}")
    print(f"  Hardest tokens (lowest accuracy):")
    for tid, acc in zip(bot_ids, bot_vals.tolist()):
        tok = tokenizer.convert_ids_to_tokens(tid)
        print(f"    {tok!r:20s}  acc={acc:.3f}")


def save_checkpoint(model, optimizer, scaler, diffusion, epoch, path):
    """Save full training state so a run can be resumed after a crash or disconnect."""
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "diffusion_total": diffusion.total,
        "diffusion_correct": diffusion.correct,
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
    diffusion.correct = ckpt["diffusion_correct"].to(device)
    return ckpt["epoch"] + 1


def train(model, diffusion, dataloader, config, resume_path=None):
    """Train the model with diffusion masking and partial reveal simulation."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0, weight_decay=config.weight_decay
    )
    use_amp = config.device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    data_iter = iter(dataloader)

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
            x_t, is_masked = diffusion.corrupt(tokens, t)

            # Reveal some masked tokens to simulate a partially decoded sequence state.
            x_t, train_mask = apply_partial_reveal(x_t, tokens, is_masked, reveal_prob=0.5)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x_t, t, key_padding_mask=pad_mask)

                # Compute loss only on tokens that are still masked and not padding.
                loss_mask = (train_mask & ~pad_mask).bool()
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
            # saw as masked (post partial-reveal), not the pre-reveal mask.
            diffusion.update(logits.detach(), tokens, train_mask)

            if loss is not None:
                running_loss += loss.item() * config.accum_steps
            loop.set_postfix(loss=f"{running_loss / (step + 1):.4f}")

        # Save model and print statistics periodically.
        if epoch % config.save_every_epochs == 0 or epoch == config.num_epochs - 1:
            save_checkpoint(model, optimizer, scaler, diffusion, epoch, f"masked_diffusion_epoch_{epoch}.pt")
            print(f"\n--- Epoch {epoch} Token Difficulty ---")
            print_token_stats(diffusion, config.tokenizer)
        print()
