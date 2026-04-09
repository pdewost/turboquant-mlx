<div align="center">
  <img src="https://img.shields.io/badge/Apple_Silicon-Optimized-000000?style=for-the-badge&logo=apple" alt="Apple Silicon"/>
<img src="https://img.shields.io/badge/Compression-KV_Cache-10b981?style=for-the-badge" alt="KV Cache"/>
<img src="https://img.shields.io/badge/Performance-5x_Savings-brightgreen?style=for-the-badge" alt="Savings"/>
  <h1>🚀 TurboQuant-MLX</h1>
  <p><b>Extreme KV Cache Compression (1-3 bit) for LLMs natively on Apple Silicon.</b></p>
</div>

> 🍏 **Part of the Mac AI Ecosystem Initiative**

## 💡 Overview
TurboQuant-MLX is a high-performance quantization framework designed to minimize the memory footprint of Large Language Models (LLMs) on macOS. By compressing the KV cache down to 1-3 bits using asymmetric PolarQuant, it allows for massive context windows on standard consumer Mac hardware.

## ✨ Key Features
- **PolarQuant Caching**: SOTA asymmetric quantization for KV cache blocks.
- **Unified Memory Optimization**: Zero-copy data movement between CPU and GPU.
- **OpenAI Compatibility**: Drop-in server replacement for high-throughput inference.

## 🔬 Under the Hood
TurboQuant utilizes specialized Metal kernels for dequantization on-the-fly, ensuring that the GPU memory bandwidth is the only limit, not the capacity.


---

## 🍏 The Mac AI Ecosystem
This initiative aims to build a world-class, high-performance AI toolkit natively for Apple Silicon.

- [🍏 **Env-Selector-MLX**](https://github.com/helgklaizar/env-selector-mlx) — UI configurator for your AI environment.
- [🌉 **Cuda-Bridge-MLX**](https://github.com/helgklaizar/cuda-bridge-mlx) — Run CUDA-dependent projects natively.
- [🚀 **TurboQuant-MLX**](https://github.com/helgklaizar/turboquant-mlx) — Extreme KV Cache Compression (1-3 bit).
- [🔥 **Flamegraph-MLX**](https://github.com/helgklaizar/flamegraph-mlx) — Energy & Performance Visual Profiler.
- [🧠 **Rag-Indexer-MLX**](https://github.com/helgklaizar/rag-indexer-mlx) — Native system RAG with zero battery drain.
- [⚒️ **Forge-MLX**](https://github.com/helgklaizar/forge-mlx) — Fast memory-efficient Fine-Tuning.
- [🔳 **BitNet-MLX**](https://github.com/helgklaizar/bitnet-mlx) — Native Ternary (1.58-bit) Kernels.
- [👁️ **OmniParser-MLX**](https://github.com/helgklaizar/omni-parser-mlx) — High-speed visual GUI understanding.
- [⚡️ **Flash-Attention-MLX**](https://github.com/helgklaizar/flash-attention-mlx) — Native FA3 for Metal.
- [🌿 **SageAttention-MLX**](https://github.com/helgklaizar/sage-attention-mlx) — Ultra-fast Quantized Attention.
- [🧬 **Attention-Matching-MLX**](https://github.com/helgklaizar/attention-matching-mlx) — Recursive 50x context compression.
- [🚀 **RocketKV-MLX**](https://github.com/helgklaizar/rocket-kv-mlx) — Extreme 400x cache pruning.
- [📡 **KVTC-MLX**](https://github.com/helgklaizar/kvtc-mlx) — Transform Coding for KV cache.
- [🌌 **AETHER-MLX**](https://github.com/helgklaizar/aether-mlx) — Geometric Sparse Attention.
- [🌌 **DeepSeek-MLX**](https://github.com/helgklaizar/deepseek-mlx) — High-throughput inference engine.
- [🎞 **Open-Sora-MLX**](https://github.com/helgklaizar/open-sora-mlx) — Text-to-Video generation pipeline.
- [🗣 **Moshi-Voice-MLX**](https://github.com/helgklaizar/moshi-voice-mlx) — Realtime Voice-to-Voice agents.
- [🎲 **MCTS-RL-MLX**](https://github.com/helgklaizar/mcts-rl-mlx) — Parallel reasoning at scale.

