# AdaMask

A prototype masked diffusion-style language model with adaptive difficulty masking.

This project implements a transformer that predicts masked tokens from partially corrupted sequences. The current training objective is closer to a masked language modeling (MLM) objective than a full autoregressive language model.

## Structure

- `main.py` - entrypoint for full-scale training
- `sanity_check.py` - fast, small-scale end-to-end pipeline check
- `generate.py` - generate and decode text from a trained checkpoint
- `adamask/config.py` - model and training configuration
- `adamask/data.py` - dataset loading and tokenization
- `adamask/diffusion.py` - masking and difficulty tracking
- `adamask/model.py` - transformer denoiser
- `adamask/train.py` - training loop and loss computation
- `adamask/sample.py` - iterative confidence-based sampling with self-revision

## Getting started

Install required dependencies:

```bash
pip install torch transformers datasets tqdm
```

Run the sanity check first (small model, a few minutes on a GPU) to confirm
the pipeline works before committing to a full-scale run:

```bash
python sanity_check.py
```

Run full-scale training:

```bash
python main.py
```

## Datasets

Two different datasets are used for two different purposes:

- **Sanity check** (`sanity_check.py`) uses [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories),
  a small (~2.1M train / ~22k validation rows), clean, plain-prose dataset of
  simple generated stories. It was chosen over `wikitext-2-raw-v1` (the
  previous default) because wikitext's raw text keeps wiki markup artifacts
  (`= = = Heading = = =` section headers, `@-@`-joined punctuation) that add
  noise without adding signal for a quick "does the pipeline learn anything"
  check.
- **Full-scale training** (`main.py`) uses `HuggingFaceFW/fineweb-edu`
  (`sample-10BT`), streamed rather than downloaded up front. This is the
  dataset the architecture is actually meant to learn from at scale.

## Notes

- The current design uses `roberta-base` as the tokenizer.
- The training objective is masked-token prediction, which is akin to an MLM rather than a standard autoregressive SLM.
- This refactor splits the code into modules so the model, diffusion logic, data pipeline, and training loop are easier to modify.

## How the code works

- `main.py` / `sanity_check.py` build a `Config`, load data, construct the model, and start training.
- `adamask/data.py` turns raw text into fixed-length token blocks with padding.
- `adamask/diffusion.py` defines the masking schedule and an adaptive difficulty tracker.
- `adamask/model.py` defines a transformer that predicts token logits from masked input plus a timestep embedding.
- `adamask/train.py` trains the model by masking tokens according to the diffusion
  schedule at a randomly sampled timestep `t`, then computing loss on exactly the
  masked positions -- matching what `adamask/sample.py` does at generation time,
  so a given `t` means the same amount of visible corruption during training and
  sampling.
- `adamask/sample.py` uses iterative unmasking to generate full sequences from the
  trained model, and can revise its own earlier guesses: each step, positions
  the model is no longer confident about get remasked and reconsidered in a
  later step with more context, instead of every revealed token being frozen
  the instant it's first filled in. This is the key structural difference from
  an autoregressive model, which can never revisit a token once generated.
  Remasking is capped and decays to zero by the final step, so generation is
  still guaranteed to finish fully unmasked within `config.steps` steps. Tune
  it with `--remask-threshold` (how unconfident the model must be in a
  position's current token before it's a remask candidate) and
  `--max-remask-frac` (the most that can be remasked in a single step) in
  `generate.py`.


