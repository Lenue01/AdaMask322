"""Inference sampler for the AdaMask model."""

import torch
import torch.nn.functional as F


def sample(model, diffusion, config, num_samples=4, temperature=1.0,
           remask_threshold=0.2, max_remask_frac=0.1):
    """Generate sequences by iteratively unmasking and refining positions.

    This starts with all tokens masked, then each step predicts logits for
    the full sequence and does two things:

    1. Reveals new tokens for still-masked positions, picking the most
       confident ones first, down to the mask count `diffusion`'s cosine
       schedule expects at the next timestep (instead of spreading reveals
       evenly over the remaining steps, which would drift from what `t` is
       supposed to mean).
    2. Remasks already-revealed positions the model is no longer confident
       about -- i.e. its predicted probability for the token currently
       sitting there has dropped below `remask_threshold` -- sending them
       back to MASK so a later step, with more of the sequence filled in
       around them, gets to reconsider them with better context. This is
       what lets the model revise its own earlier guesses instead of
       freezing them the instant they're first revealed, which a causal
       autoregressive model structurally cannot do.

    Remasked positions are excluded from *this same step's* reveal pool, so
    reconsideration actually waits for a fresh forward pass with updated
    context on the next step, rather than being immediately overwritten by
    the very prediction that just flagged it as low-confidence. Remasking
    is disabled on the final step and its per-step budget (`max_remask_frac`
    of currently-revealed positions) decays as the schedule progresses, so
    the sequence is still guaranteed to end fully unmasked after
    `config.steps` steps: the last step's target mask count is always 0,
    forcing anything still masked to be revealed.
    """
    model.eval()
    device = config.device
    L = config.context_length
    x = torch.full((num_samples, L), config.mask_token_id, device=device)
    mask = torch.ones(num_samples, L, dtype=torch.bool, device=device)
    num_steps = config.steps

    for step in range(num_steps):
        if mask.sum() == 0:
            break
        t_val = max(1, num_steps - step)
        t = torch.full((num_samples,), t_val, device=device, dtype=torch.long)
        logits = model(x, t)
        step_temp = temperature * (0.5 + 0.5 * t_val / num_steps)
        probs = F.softmax(logits / step_temp, dim=-1)
        pred_tokens = torch.multinomial(probs.view(-1, config.vocab_size), 1).view(num_samples, L)
        confidence = probs.max(dim=-1).values

        # Remask already-revealed positions the model no longer endorses.
        # Skipped on the final step (t_val == 1) so every position is still
        # guaranteed to end up filled by the time the loop ends.
        just_remasked = torch.zeros_like(mask)
        if t_val > 1:
            self_conf = probs.gather(-1, x.unsqueeze(-1)).squeeze(-1)
            budget_frac = max_remask_frac * (t_val / num_steps)
            for i in range(num_samples):
                revealed_pos = (~mask[i]).nonzero(as_tuple=True)[0]
                if revealed_pos.numel() == 0:
                    continue
                candidates = revealed_pos[self_conf[i][revealed_pos] < remask_threshold]
                if candidates.numel() == 0:
                    continue
                cap = int(round(budget_frac * revealed_pos.numel()))
                if cap == 0:
                    continue
                if candidates.numel() > cap:
                    worst = self_conf[i][candidates].topk(cap, largest=False).indices
                    candidates = candidates[worst]
                x[i, candidates] = config.mask_token_id
                mask[i, candidates] = True
                just_remasked[i, candidates] = True

        # Reveal down to the mask count the schedule expects at the next
        # timestep, rather than an even split of the remaining steps.
        t_next = t_val - 1
        target_count = int(round(diffusion.mask_rate(t_next).item() * L))
        remaining = mask.sum(dim=1)
        k_per_sample = (remaining - target_count).clamp(min=0)
        confidence_masked = confidence.clone()
        confidence_masked[~mask] = -1.0
        for i in range(num_samples):
            # Positions remasked earlier in this same step are excluded here
            # so they wait for next step's fresh forward pass instead of
            # being immediately re-filled by the pass that just doubted them.
            masked_pos = (mask[i] & ~just_remasked[i]).nonzero(as_tuple=True)[0]
            if masked_pos.numel() == 0:
                continue
            k_i = min(k_per_sample[i].item(), masked_pos.numel())
            if k_i == 0:
                continue
            conf_masked = confidence_masked[i][masked_pos]
            _, top_local = conf_masked.topk(k_i)
            chosen = masked_pos[top_local]
            x[i].scatter_(0, chosen, pred_tokens[i].gather(0, chosen))
            mask[i][chosen] = False
    return x
