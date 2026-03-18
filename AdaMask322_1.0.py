"""
Masked diffusion Language model


        Key Processes:
        1. In the Forward process tokens are replaces with [MASK] 
        2. A transformer model denoises the masked sample
        3. To Sample, start with all MASK tokens and denoise from t = 100 to t = 0

        My Original model was a guasian noise diffusion model that added noise to token embeddings and used a transformer to guess the original embedding
        This does not work for text embeddings

        After further research into the subject I learned about masked diffusion models.  This is my attempt to impliment a masked diffusion model
        

        

"""

#---------------Imports-----------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
#----------End of imports----------------



#------------------Tokenizer-------------------
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
#------------------Tokenizer-------------------



#------------------Config----------------------
class config:
    vocab_size = len(tokenizer)
    mask_token_ID = tokenizer.mask_token_id
    pad_token_ID = tokenizer.pad_token_id
    context_length = 128
    hidden_size = 1024
    heads = 16
    layers = 16
    steps = 64
    batch_size = 128
    lr = 1e-4
    weight_decay = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"
#----------------Config-------------------------



#-------------------data-----------------------------
def get_dataloader(tokenizer, config):
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    dataset = dataset.shuffle(seed=42)

    def tokenize(example):
        enc = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=config.context_length,
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    dataset = dataset.map(tokenize, num_proc=4, remove_columns=dataset.column_names)
    dataset = dataset.with_format("torch")
    return DataLoader(dataset, batch_size=config.batch_size, num_workers=4,
                      pin_memory=True, shuffle=True)
#-------------------data----------------------------

class MaskedDiffusion:
    def __init__(self, steps, masked_token_ID, device):
        self.steps = steps
        self.masked_token_ID = masked_token_ID
        self.device = device

        #t is tensor of length steps
        t = torch.arange(steps + 1, dtype = torch.float32) 
        '''1 - alpha or a(t) is the chance of a token being masked. a(t) = .7 
        means there is a 70% chance of token being masked
        a(0) = means 0% chance
        a(T) = 100% chance
        '''
        alpha = ((torch.cos(math.pi/2 * t/self.steps))**2)
        """
        Divide by a[0] to normalize the timestamps
        """
        alpha = alpha/alpha[0]
        self.alpha = alpha.to(device)

    def mask_rate(self, t):
        return 1.0 - self.alpha[t]


    def corrupt(self, tokens, t):
            '''
            Get the 1 - alpha for mask rate
            '''
            rate = self.mask_rate(t).view(-1, 1)

            '''Generate probabilities in same shape of tokens'''
            noise = torch.rand(tokens.shape, device=tokens.device)
            '''Is masked if the noise is less than the mask rate and the token isnt padding'''
            is_masked = (noise < rate) & (tokens != config.pad_token_ID)
            '''Make a clone of the original token'''
            x_t = tokens.clone()
            ''''
            For every index where a token is masked replace the token with mask token ID
            '''
            x_t[is_masked] = self.masked_token_ID
            
            '''
            Retunrns x_t (the corrupted tokens) and is_masked a tesnsor map
            of the masked and unmasked tokens
            '''
            return x_t, is_masked
        

class TokenDifficulty(MaskedDiffusion):
    def __init__(self, vocab_size, masked_token_ID, steps, device):
        super().__init__(steps, masked_token_ID, device)

        self.total   = torch.ones(vocab_size, device=device)
        self.correct = torch.full((vocab_size,), 0.5, device=device)

    def update(self, logits, tokens, is_masked):
        '''
        logits: predicted tokens
        tokens: original tokens
        is_maksed: true or false
        
        every time the model predicts a masked token

        count it as seen

        check if prediction was right

        update the counts

        '''

        target = tokens[is_masked]
        probs = F.softmax(logits[is_masked].float(), dim=-1)
        correct_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)  # P(correct token)

        self.correct *= 0.99
        self.total   *= 0.99
        self.total.scatter_add_(0, target, torch.ones_like(target, dtype=torch.float))
        self.correct.scatter_add_(0, target, correct_probs)

    def get_difficulty(self, tokens):
        return 1.0 - (self.correct[tokens] / (self.total[tokens] + 1e-8))
    
    def difficulty_corrupt(self, tokens, t):
        rate = self.mask_rate(t).view(-1, 1)          # [B, 1] target fraction
        difficulty = self.get_difficulty(tokens)       # [B, L] scores in [0,1]
        # Bias noise downward for hard tokens → they're more likely to be masked
        noise = torch.rand(tokens.shape, device=tokens.device)
        biased = noise * (1.0 - 0.3 * difficulty)     # hard tokens get lower threshold
        is_masked = (biased < rate) & (tokens != config.pad_token_ID)
        x_t = tokens.clone()
        x_t[is_masked] = self.masked_token_ID
        return x_t, is_masked

    



def timestamp_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args  = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)



class AdaLNBlock(nn.Module):
    """
    Transformer block with Adaptive Layer Norm (AdaLN) conditioning.
    The diffusion timestep t modulates scale and shift of each LN,
    letting every layer adapt its behavior based on how noisy the input is.
    """

    def __init__(self, config):
        super().__init__()
        H = config.hidden_size

        self.ln_1 = nn.LayerNorm(H, elementwise_affine=False)
        self.ln_2 = nn.LayerNorm(H, elementwise_affine=False)

        self.attn = nn.MultiheadAttention(H, config.heads, batch_first=True, dropout=0.0)

        self.mlp = nn.Sequential(
            nn.Linear(H, H * 4),
            nn.GELU(),
            nn.Linear(H * 4, H),
            nn.Dropout(0.1),
        )

        # Projects t_emb into (shift_attn, scale_attn, shift_mlp, scale_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(H, 4 * H),
        )


    def forward(self, x, t_emb, key_padding_mask=None):
        """
        Args:
            x               : [B, L, H]
            t_emb           : [B, 1, H]  timestep conditioning
            key_padding_mask: [B, L]     True at pad positions (ignored in attention)
        """
        mod = self.adaLN_modulation(t_emb)                        # [B, 1, 4H]
        shift_msa, scale_msa, shift_mlp, scale_mlp = mod.chunk(4, dim=-1)

        # Attention branch
        x_mod    = self.ln_1(x) * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(x_mod, x_mod, x_mod, key_padding_mask=key_padding_mask)
        x = x + attn_out

        # MLP branch
        x_mod = self.ln_2(x) * (1 + scale_mlp) + shift_mlp
        x = x + self.mlp(x_mod)

        return x
    

class MaskedDiffusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        H = config.hidden_size


        #Takes the token and converts it to an embedding of H size 
        self.token_emb = nn.Embedding(config.vocab_size, H) 
        
        #Tells the model where tokens appear in a sequence
        self.pos_emb   = nn.Parameter(torch.randn(1, config.context_length, H) * 0.02)

        
        self.time_mlp = nn.Sequential(
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )


        self.blocks = nn.ModuleList([AdaLNBlock(config) for _ in range(config.layers)])

        self.ln_f = nn.LayerNorm(H)
        self.head = nn.Linear(H, config.vocab_size, bias=False)

        self._init_weights()

        # Weight tying: share embedding ↔ unembedding matrix
        self.head.weight = self.token_emb.weight

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_t, t, key_padding_mask=None):
        #embedds tokens and adds the posititinal embedding
        x = self.token_emb(x_t) + self.pos_emb[:, :x_t.size(1)]

        # Timestep conditioning
        
        t_emb = timestamp_embedding(t, x.size(-1))         # [B, H]
        t_emb = self.time_mlp(t_emb).unsqueeze(1)         # [B, 1, H]

        #passes the token embeddings through the transfoermer layers
        for block in self.blocks:
            x = block(x, t_emb, key_padding_mask=key_padding_mask)

        #normalize embeddings before prediction 
        hidden = self.ln_f(x)                             # [B, L, H]
        
        #prediction
        logits = self.head(hidden)                         # [B, L, V]
        return logits
    


@torch.no_grad()
def sample(model, config, num_samples=4, temperature=1.0):
    """
    Iterative confidence-based unmasking.

    Algorithm:
      1. Start with all tokens = [MASK]
      2. At each step t (counting down from train_cap → 1):
         a. Apply repetition penalty to logits for already-placed tokens
         b. Sample token predictions from the distribution (multinomial)
         c. Randomly select k masked positions to permanently unmask
      3. Return the fully unmasked sequence

    Random unmasking order avoids the confidence feedback loop that
    causes repetitive outputs in confidence-based (MaskGIT-style) sampling.
    """
    model.eval()
    device = config.device

    # Start fully masked
    x    = torch.full((num_samples, config.context_length), config.mask_token_ID, device=device)
    mask = torch.ones(num_samples, config.context_length, dtype=torch.bool, device=device)

    # How many tokens to unmask per step (evenly distributed)
    max_t = int(config.steps * 0.65)
    tokens_per_step = math.ceil(config.context_length / max_t)


    for step in range(max_t):
        t_val = max(1, max_t - step)               # counts down from train cap → 1
        t     = torch.full((num_samples,), t_val, device=device, dtype=torch.long)

        logits = model(x, t)

        # Apply repetition penalty: reduce prob of tokens already placed
        logits_penalized = logits.clone()
        for i in range(num_samples):
            placed = x[i][~mask[i]].unique()                                      # unique placed tokens only
            if placed.numel() > 0:
                logits_penalized[i, :, placed] -= 1.5                             # flat penalty per token type
        probs       = F.softmax(logits_penalized / temperature, dim=-1)           # [B, L, V]
        pred_tokens = torch.multinomial(                                           # [B, L]
            probs.view(-1, probs.size(-1)), 1
        ).view(num_samples, -1)

        if mask.sum() == 0:
            break

        for i in range(num_samples):
            masked_pos = mask[i].nonzero(as_tuple=True)[0]
            if masked_pos.numel() == 0:
                continue
            k_i = min(tokens_per_step, masked_pos.numel())
            chosen = masked_pos[torch.randperm(masked_pos.numel(), device=device)[:k_i]]
            x[i].scatter_(0, chosen, pred_tokens[i].gather(0, chosen))
            mask[i].scatter_(0, chosen, torch.zeros(k_i, dtype=torch.bool, device=device))

    return x



def print_token_stats(diffusion, tokenizer, k=10):
    accuracy = diffusion.correct / (diffusion.total + 1e-8)   # [vocab_size]
    seen_mask = diffusion.total > 1.0                          # only tokens seen enough

    seen_acc = accuracy[seen_mask]
    seen_ids = seen_mask.nonzero(as_tuple=True)[0]

    if seen_ids.numel() < k:
        print("  Not enough token data yet.")
        return

    top_vals, top_idx  = seen_acc.topk(k)
    bot_vals, bot_idx  = seen_acc.topk(k, largest=False)

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


def get_lr(global_step, warmup_steps=2000, max_lr=config.lr, min_lr=config.lr/10, total_steps=1000000):
    if global_step < warmup_steps:
        return max_lr * global_step / warmup_steps
    progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(model, diffusion, dataloader, config, tokenizer):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0, weight_decay=config.weight_decay
    )

    use_amp = config.device == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    STEPS_PER_EPOCH = 8000
    ACCUM_STEPS     = 1
    data_iter       = iter(dataloader)

    for epoch in range(100):
        model.train()
        running_loss = 0.0
        loop = tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch}")
        for step in loop:
            try:
                batch = next(data_iter)
                tokens = batch['input_ids'].to(config.device)
                attn_msk = batch["attention_mask"].to(config.device)
                pad_mask = (attn_msk == 0)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
                tokens   = batch['input_ids'].to(config.device)
                attn_msk = batch["attention_mask"].to(config.device)
                pad_mask = (attn_msk == 0)

            global_step = epoch * STEPS_PER_EPOCH + step
            lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if step % ACCUM_STEPS == 0:
                optimizer.zero_grad(set_to_none=True)

            #decide either to do random masking or difficulty masking
            split = torch.randint(0, 10, (1,)).item()

            t = torch.randint(1, int(config.steps * 0.65) + 1, (tokens.size(0),), device=config.device)
            device_type = "cuda" if config.device == "cuda" else "cpu"

            if split < 7:
                x_t, is_masked = diffusion.corrupt(tokens, t)
            else:
                x_t, is_masked = diffusion.difficulty_corrupt(tokens, t)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x_t, t, key_padding_mask=pad_mask)

                # Compute CE loss only on masked positions (ignore padding + unmasked)
                loss_mask = (is_masked & ~pad_mask).bool()             # [B, L]
                if not loss_mask.any():
                    continue
                loss = F.cross_entropy(
                    logits[loss_mask],                         # [N_masked, V]
                    tokens[loss_mask],                         # [N_masked]
                    label_smoothing=0.1,
                ) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            diffusion.update(logits.detach(), tokens, is_masked)
            running_loss += loss.item() * ACCUM_STEPS   # display unscaled loss
            loop.set_postfix(loss=f"{running_loss / (step + 1):.4f}") 


        if epoch % 5 == 0 or epoch == 99:
            torch.save(model.state_dict(), f"masked_diffusion_epoch_{epoch}.pt")
            print(f"\n--- Epoch {epoch} Token Difficulty ---")
            print_token_stats(diffusion, tokenizer)
            print(f"\n--- Epoch {epoch} Samples ---")
            model.eval()
            with torch.no_grad():
                seqs = sample(model, config, num_samples=3, temperature=1.2)
            for i, seq in enumerate(seqs):
                print(f"  [{i}] {tokenizer.decode(seq, skip_special_tokens=True)}")
            model.train()
        print()

if __name__ == "__main__":
    diffusion  = TokenDifficulty(config.vocab_size, config.mask_token_ID, config.steps, config.device)
    model      = MaskedDiffusionTransformer(config).to(config.device)
    dataloader = get_dataloader(tokenizer, config)
    train(model, diffusion, dataloader, config, tokenizer)