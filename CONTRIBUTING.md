# Contributing to Mac AI Ecosystem Initiative

First off, thank you for considering contributing to our initiative! It's people like you that make the Apple Silicon AI community such a great place.

## 🌈 Our Mission
We are building a highly-optimized, native, and aggressive AI stack specifically for Apple Silicon (M1-M5). Our goal is to bring the best of the CUDA world to Metal/MLX and beyond.

## 🛠 How Can I Contribute?

### Reporting Bugs
* Check the Issues tab to see if the bug has already been reported.
* If not, create a new issue. Include a clear title, a description of the problem, and steps to reproduce.

### Suggesting Enhancements
* We love new ideas! If you have a suggestion for improving a specific project or the ecosystem as a whole, open an issue labeled `enhancement`.

### Pull Requests
1. **Fork the repo** and create your branch from `main`.
2. **Ensure your code follows the style** of the existing codebase. We prefer clean, modular, and well-documented code.
3. **Optimized for Mac**: Every kernel or intensive operation must be benchmarked on Apple Silicon. We do not accept unoptimized PyTorch generic code if a Metal/MLX version is possible.
4. **Issue a PR** with a clear description of your changes.

## 📜 Coding Standards
* **Backend**: Python 3.11+, MLX, Rust (for systems), or C++ (for Metal kernels).
* **Frontend**: Vite + React + Tailwind (if applicable).
* **Documentation**: Every PR should update the README if necessary.

## ⚖️ License
By contributing, you agree that your contributions will be licensed under the **MIT License**.

---
*Stay hungry. Stay foolish. Optimize for Metal.* 🍏
