# Question-in-the-Loop: Interleaved Query Tokens as Visual Summarizers for Video LLMs
The authors' implementation on LLaVA-Video-7B-Qwen2 (ACM MM 2026).

## Quick start

```bash
conda create -n qitl python=3.10 -y && conda activate qitl
pip install -r requirements.txt
```
## Faster runs on long videos: the frame cache (optional)

On Video-MME, MLVU and LongVideoBench, decoding 64 frames from a video up to an hour long costs more CPU time than the model spends on the GPU. 

The frame cache stores the exact tensors the model receives for each video, about 27 MB per video. Set `LLAVA_VID_FRAME_CACHE`, and the first run on a benchmark fills the cache as it decodes. Later runs read from it instead of decoding. 

```bash
export LLAVA_VID_FRAME_CACHE=/data/qitl_frame_cache
RETENTION_RATIO=0.25 TASKS=videomme bash scripts/eval.sh   # decodes and fills the cache
RETENTION_RATIO=0.1  TASKS=videomme bash scripts/eval.sh   # reads the cache
```
The cache for all three long-video benchmarks takes about 80 GB. You can delete it when you are done.

## Running evaluations

```bash
RETENTION_RATIO=0.25 TASKS=mvbench bash scripts/eval.sh                  # QitL, keep 25%
RETENTION_RATIO=0.1  TASKS=videomme bash scripts/eval.sh                 # QitL, keep 10%
```
Scores are printed and saved, with every answer, under `results/`.

| Variable | Default | Meaning |
|---|---|---|
| `RETENTION_RATIO` | `0.25` | Share of image patches kept in every frame. `1.0` runs the original model. |
| `TASKS` | `mvbench` | Comma-separated task names: `mvbench`, `videomme`, `longvideobench_val_v`, `mlvu_dev`. |
| `MODEL_PATH` | `lmms-lab/LLaVA-Video-7B-Qwen2` | Hugging Face id or local checkpoint folder. |
| `OUTPUT_DIR` | `results/<tasks>_r<ratio>` | Scores (`*_results.json`) and every answer (`*_samples_*.jsonl`). |
| `LLAVA_VID_LOADER_WORKERS` | `8` | CPU threads that decode upcoming videos while the GPU works. |
| `LLAVA_VID_FRAME_CACHE` | unset | Folder where runs store and reuse decoded frames. |

## Code map

| Path | What it is |
|---|---|
| `LLaVA-NeXT/llava/model/language_model/qitl.py` | The method: question copies, attention rules, token selection, pruned forward pass. |
| `LLaVA-NeXT/llava/model/llava_arch.py` | Inserts the question copies while building the input (marked `qitl`). |
| `LLaVA-NeXT/llava/model/language_model/llava_qwen.py` | Sends the first forward pass to QitL; generation then continues as usual. |
| `lmms-eval/lmms_eval/models/simple/llava_vid.py` | Evaluation wrapper. |
| `scripts/eval.sh` | Evaluation launcher. |

`LLaVA-NeXT/` is based on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), and `lmms-eval/` on [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) v0.6.1.

## License and acknowledgements

The QitL code is released under the Apache 2.0 license (`LICENSE`). It is built on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) (Apache 2.0) and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) (MIT)
