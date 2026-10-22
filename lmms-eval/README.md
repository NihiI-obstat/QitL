# lmms-eval (reduced copy)

This is [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) v0.6.1 (MIT licence, see `LICENSE`), cut down for the QitL release:

- Only the task definitions of MVBench, Video-MME, LongVideoBench and MLVU are kept, with per-category scores added for MVBench and Video-MME.
- `lmms_eval/models/simple/llava_vid.py` adds the `retention_ratio` argument, background video loading, an optional frame cache and stage timing.

See the top-level `README.md` for setup and usage, and the upstream repository for the full toolkit and its documentation.
