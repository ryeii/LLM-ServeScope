# LLM-ServeScope production deployment

This branch contains the prebuilt GitHub Pages edition. Source code, purpose,
limitations, setup and full documentation are on the [main branch](https://github.com/ryeii/LLM-ServeScope/tree/main).

LLM-ServeScope is a behavioral simulation of LLM serving backed by SimPy. It
estimates throughput, latency, memory, energy and utilization, with per-GPU
performance and tokens per Joule. It is **behavioral, not mechanistic simulation**.

GPU timing uses separately fitted KAIST BF16 and Microsoft Vidur FP16 profiles.
See the [cross-validated accuracy report](https://github.com/ryeii/LLM-ServeScope/blob/main/docs/accuracy-2026-10-19/README.md)
for overall, per-operator and unseen-GPU residuals. Operator accuracy does not
calibrate serving throughput, energy or network behavior.

The same Python model runs in a Pyodide browser worker. The compiled native
LLMServingSim/ASTRA-Sim reference engine requires the local installation.

LLM-ServeScope design and implementation: [ryeii (Ryan Z An)](https://github.com/ryeii),
including the behavioral modeling extension, interface, visualizations, browser
execution layer and native backend integration.

Reference backend: LLMServingSim 2.0 by CASYS Lab at KAIST — Jaehong Cho, Hyunmin Choi,
Guseul Heo and Jongse Park. [Upstream source](https://github.com/casys-kaist/LLMServingSim).
FP16 profiling corpus: [Microsoft Vidur](https://github.com/microsoft/vidur).
Published prefix-reuse workload evidence: [Mooncake contributors](https://github.com/kvcache-ai/Mooncake).
Optional reuse fractions describe earlier-arrival potential; ready HBM availability
is an assumption, not measured hardware cache hits or end-to-end validation.
See THIRD_PARTY_NOTICES.md and licenses/ for retained credits and license texts.

Deploy this branch from production / (root) in GitHub Pages settings.
