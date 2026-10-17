# LLM-ServeScope production deployment

This branch contains the prebuilt GitHub Pages edition. Source code, purpose,
limitations, setup and full documentation are on the [main branch](https://github.com/ryeii/LLM-ServeScope/tree/main).

LLM-ServeScope is a behavioral simulation of LLM serving backed by SimPy. It
estimates throughput, latency, memory, energy and utilization, with per-GPU
performance and tokens per Joule. It is **behavioral, not mechanistic simulation**.

The same Python model runs in a Pyodide browser worker. The compiled native
LLMServingSim/ASTRA-Sim reference engine requires the local installation.

Credit: LLMServingSim 2.0 by CASYS Lab at KAIST — Jaehong Cho, Hyunmin Choi,
Guseul Heo and Jongse Park. [Upstream source](https://github.com/casys-kaist/LLMServingSim).
See THIRD_PARTY_NOTICES.md and licenses/ for retained credits and license texts.

Deploy this branch from production / (root) in GitHub Pages settings.
