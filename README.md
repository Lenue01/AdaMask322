# AdaMask

A prototype masked diffusion-style language model with adaptive difficulty masking.

This project implements a transformer that predicts masked tokens from partially corrupted sequences. The current training objective is closer to a masked language modeling (MLM) objective than a full autoregressive language model.

## Structure

- `main.py` - entrypoint for training
- `adamask/config.py` - model and training configuration
- `adamask/data.py` - dataset loading and tokenization
- `adamask/diffusion.py` - masking and difficulty tracking
- `adamask/model.py` - transformer denoiser
- `adamask/train.py` - training loop and loss computation
- `adamask/sample.py` - iterative confidence-based sampling

## Getting started

Install required dependencies:

```bash
pip install torch transformers datasets tqdm
```

Run training:

```bash
python main.py
```

## Notes

- The current design uses `roberta-base` and a sample Hugging Face dataset.
- The training objective is masked-token prediction, which is akin to an MLM rather than a standard autoregressive SLM.
- This refactor splits the code into modules so the model, diffusion logic, data pipeline, and training loop are easier to modify.
## How the code works

- `main.py` builds a `Config`, loads data, constructs the model, and starts training.
- `adamask/data.py` turns raw text into fixed-length token blocks with padding.
- `adamask/diffusion.py` defines the masking schedule and an adaptive difficulty tracker.
- `adamask/model.py` defines a transformer that predicts token logits from masked input plus a timestep embedding.
- `adamask/train.py` trains the model by masking tokens, revealing some of them, and computing loss on the remaining masked positions.
- `adamask/sample.py` uses iterative unmasking to generate full sequences from the trained model.


