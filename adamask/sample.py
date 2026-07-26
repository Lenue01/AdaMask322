"""Inference sampler for the AdaMask model."""

import torch
import torch.nn.functional as F


def sample(model, diffusion, config, num_samples=4, temperature=1.0):
    """Generate sequences by iteratively unmasking the most confident positions.

    This starts with all tokens masked, then repeatedly predicts token logits,
    selects the most confident masked positions, and fills them in until the
    sequence is fully unmasked. `diffusion` supplies the same cosine mask-rate
    schedule used to corrupt inputs during training, so the reveal count each
    step keeps the actual masked fraction consistent with the timestep `t`
    being passed to the model (instead of just spreading reveals evenly over
    the remaining steps, which drifts from what `t` is supposed to mean).
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
        confidence[~mask] = -1.0

        # Reveal down to the mask count the schedule expects at the next
        # timestep, rather than an even split of the remaining steps.
        t_next = t_val - 1
        target_count = int(round(diffusion.mask_rate(t_next).item() * L))
        remaining = mask.sum(dim=1)
        k_per_sample = (remaining - target_count).clamp(min=0)
        for i in range(num_samples):
            masked_pos = mask[i].nonzero(as_tuple=True)[0]
            if masked_pos.numel() == 0:
                continue
            k_i = min(k_per_sample[i].item(), masked_pos.numel())
            if k_i == 0:
                continue
            conf_masked = confidence[i][masked_pos]
            _, top_local = conf_masked.topk(k_i)
            chosen = masked_pos[top_local]
            x[i].scatter_(0, chosen, pred_tokens[i].gather(0, chosen))
            mask[i][chosen] = False
    return x
