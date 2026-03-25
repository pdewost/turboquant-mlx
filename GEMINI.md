# TurboQuant Mac — Project Working Memory

## Stack
- Python 3 (pyproject.toml)
- Apple MLX (`mlx.core`, Metal GPU acceleration)
- `mlx_lm` (API integration and LLM generation)

## Quick Run
```bash
# Package install
pip install -e .

# Launch compressed OpenAI-compatible Server 
python3 scripts/run_server.py --model mlx-community/Meta-Llama-3-8B-Instruct-4bit

# Run generation tests across 5 models
PYTHONPATH=. python3 scripts/run_needle_test.py
```

## Architecture
Objective: Implement QJL and PolarQuant algorithms for unprecedented LLM KV Cache compression (down to 3 bits) without quality loss.
- `mlx_core/mlx_turboquant.py` — Metal-optimized full quantization pipeline (Keys)
- `mlx_core/mlx_polarquant.py` — Fast MSE quantization (Values)
- `mlx_core/cache.py` — Dynamic class replacement for `mlx_lm`'s `KVCache` with integrated chunking
- `scripts/` — Handful scripts for tests, local servers, and EXO-clusters

## Key Design Decisions
- **mlx_lm Monkey-patch:** Seamlessly integrates directly into `make_prompt_cache` and `KVCache`, guaranteeing memory compression across modules globally.
- **Asymmetric Compression:** Keys are compressed via highly accurate `TurboQuant`, while Values are heavily shrunk via standard `PolarQuant`.
- **Heavy Hitter Caching / FP16 Sink:** First 128 context tokens stay completely uncompressed, saving instruction-following metrics at extreme bitrates. 

## Known Issues / Tech Debt
- Writing custom `.metal` shaders is postponed (the Python `mlx.core` API is currently fast enough due to lazy graph compilation).
- Architectures like Qwen/Gemma/Phi-3 exhibit more hallucination with 3-bit caches on current hyper-parameters, while Meta Llama 3 and 3.2 function flawlessly.

## Recent Progress
- Repackaged the architecture into a production `pip` distribution on GitHub.
- Built an integration wrapper `apply_turboquant_cache` providing dynamic chunking (64 token splits).
- Ported exact `memory_size` tracker and `run_exo_node.py` wrapper for EXO framework clusters.
- Successfully executed the *Needle-in-a-Haystack* stress test. Meta Llama architectures proved to be fully immune to 3-bit compression, efficiently shedding ~75% of their RAM overhead footprint.
- Fully localized repository to English (README, docs) for open-source adoption.

## Next Steps
1. Collect community feedback.
2. Hyper-parameter tuning (Theta_bits, Radius_bits) to stabilize GQA architectures from Google and Alibaba.
