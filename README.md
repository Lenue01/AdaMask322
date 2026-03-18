# AdpativeMasking322M - AdaMask322
A diffusion based LLM. Tokens are embedded and masked by a cosine scheduler.  The diffusion model takes in a sentence such as "The dog ran outside in the rain" and masks the toekns randomly.
The sentence can become "the [MASK] ran [MASK] in the [MASK]". The model then tries to guess the masked tokens.
The Adaptive Masking 322 parameter model adds in adpative difficulty masking to force the model to train on tokens it struggles with.
30% of the time instead of randomly masking tokens it masks tokens above a difficulty threshold forcing the model to learn hard tokens that may not get masked as frequently.
70% of the time the model follows the standered random masking procdedure seen in common Diffusion LLMs



