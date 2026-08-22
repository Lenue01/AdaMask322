"""Inference sampler for the AdaMask model."""

import torch
import torch.nn.functional as F


def _ngram_block_mask(token_list, revealed, ngram_size, mask_positions):
    """For each position in mask_positions whose preceding (ngram_size - 1)
    tokens are all revealed, find which candidate tokens would recreate an
    n-gram that's already present among the currently-revealed runs.

    Only contiguous revealed runs count as "already present" -- a masked gap
    breaks a run, since we don't know what goes there yet and shouldn't treat
    it as matching anything. Positions without a complete revealed prefix are
    skipped for the same reason: nothing to check against yet.

    Returns {position: set(forbidden_token_ids)}.
    """
    if ngram_size < 2:
        return {}
    L = len(token_list)
    seen_ngrams = set()
    i = 0
    while i < L:
        if not revealed[i]:
            i += 1
            continue
        j = i
        while j < L and revealed[j]:
            j += 1
        run = token_list[i:j]
        for k in range(len(run) - ngram_size + 1):
            seen_ngrams.add(tuple(run[k:k + ngram_size]))
        i = j

    if not seen_ngrams:
        return {}

    bad_next = {}
    for ngram in seen_ngrams:
        bad_next.setdefault(ngram[:-1], set()).add(ngram[-1])

    blocked = {}
    for p in mask_positions:
        prefix_start = p - (ngram_size - 1)
        if prefix_start < 0 or not all(revealed[prefix_start:p]):
            continue
        forbidden = bad_next.get(tuple(token_list[prefix_start:p]))
        if forbidden:
            blocked[p] = forbidden
    return blocked


def sample(model, diffusion, config, num_samples=4, temperature=1.0,
           remask_threshold=0.2, max_remask_frac=0.1,
           repetition_penalty=0.5, repetition_cap=3, no_repeat_ngram_size=3):
    """Generate sequences by iteratively unmasking and refining positions.

    This starts with all tokens masked, then each step predicts logits for
    the full sequence and does four things:

    1. Discounts each token's logit by how many times it already appears
       elsewhere in the sequence (capped at `repetition_cap` occurrences, so
       ordinary words that legitimately repeat a lot in normal text -- the,
       and, of -- aren't punished into oblivion; only repetition past the
       cap keeps getting suppressed). Without this, confidence-based reveal
       always favors repeating whatever's already locally common, since a
       repeat is definitionally the lowest-entropy guess -- which spirals
       into runaway loops (the same word revealed dozens of times) that get
       more confident the longer they run, since remasking below can't
       catch a high-confidence mistake.
    2. Hard-blocks any candidate token that would recreate a `no_repeat_ngram_size`
       n-gram already present in the sequence (the same idea as
       `no_repeat_ngram_size` in standard text-generation libraries). This is
       a different failure mode from (1): a short phrase built from ordinary,
       individually-common words (e.g. "put it in her pocket") can repeat
       verbatim without any single token in it ever tripping the per-token
       repetition penalty, since none of those words needs to be locally rare
       to make the phrase itself feel like a repeat.
    3. Reveals new tokens for still-masked positions, picking the most
       confident ones first, down to the mask count `diffusion`'s cosine
       schedule expects at the next timestep (instead of spreading reveals
       evenly over the remaining steps, which would drift from what `t` is
       supposed to mean).
    4. Remasks already-revealed positions the model is no longer confident
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

        if repetition_penalty > 0:
            valid = x != config.mask_token_id
            token_ids = x.clone()
            token_ids[~valid] = 0
            counts = torch.zeros(num_samples, config.vocab_size, device=device)
            counts.scatter_add_(1, token_ids, valid.float())
            # No penalty for the first `repetition_cap` legitimate occurrences
            # (so "the"/"and"/"of" aren't punished), then penalty grows without
            # bound past it -- clamping the penalty itself at the cap (as this
            # used to do) makes it a fixed, eventually-beatable ceiling: once a
            # token passes repetition_cap occurrences the penalty stops
            # increasing no matter how many more times it repeats, so a token
            # the model is confident enough about can out-bid that fixed
            # deterrent and spiral into an unbounded repeat loop.
            excess = (counts - repetition_cap).clamp(min=0)
            logits = logits - repetition_penalty * excess.unsqueeze(1)

        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            NEG_INF = -1e9
            token_lists = x.tolist()
            revealed_lists = (~mask).tolist()
            for i in range(num_samples):
                mask_positions = mask[i].nonzero(as_tuple=True)[0].tolist()
                blocked = _ngram_block_mask(token_lists[i], revealed_lists[i], no_repeat_ngram_size, mask_positions)
                for p, forbidden_tokens in blocked.items():
                    idx = torch.tensor(list(forbidden_tokens), device=device, dtype=torch.long)
                    logits[i, p, idx] = NEG_INF

        step_temp = temperature * (0.5 + 0.5 * t_val / num_steps)
        probs = F.softmax(logits / step_temp, dim=-1)
        pred_tokens = torch.multinomial(probs.view(-1, config.vocab_size), 1).view(num_samples, L)
        # Confidence must reflect the token we're actually about to commit
        # (the multinomial sample), not the argmax -- at temperature > 0 those
        # can differ, and ranking reveals by argmax-confidence while writing a
        # different, lower-probability sampled token defeats the point of
        # confidence-ordered revealing (commit to unambiguous tokens first).
        confidence = probs.gather(-1, pred_tokens.unsqueeze(-1)).squeeze(-1)

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
