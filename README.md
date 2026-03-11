# CactusLLM
A diffusion based LLM. Tokens are embedded and corrupted by a cosine scheduler. A transformer model takes in the corrupted embeddings and tries to predict the clean embedding. By doing so it learns the statistical layout and rules of language. It generates an entire sequence and denoises it sequentially, unlike autoregressive models.
