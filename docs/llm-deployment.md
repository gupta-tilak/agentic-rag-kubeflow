# Deploying the LLM on Kubernetes with KServe

This guide walks through deploying an open-source LLM as a KServe
InferenceService and wiring it into the agentic-RAG system.

---

## Architecture

```
┌──────────────┐    /v1/chat/completions    ┌──────────────────┐
│  RAG Agent   │ ────────────────────────►  │  llm-server      │
│  (KServe)    │    OpenAI-compatible API    │  (vLLM on GPU)   │
└──────────────┘                            └──────────────────┘
        │                                          │
        │  /query (Chroma)                         │  HuggingFace Hub
        ▼                                          ▼
┌──────────────┐                            ┌──────────────────┐
│  ChromaDB    │                            │  Model Weights   │
└──────────────┘                            └──────────────────┘
```

The **RAG agent** InferenceService calls the **llm-server**
InferenceService over the cluster network using the standard
OpenAI `/v1/chat/completions` endpoint exposed by vLLM.

---

## Prerequisites

| Requirement          | Why                                          |
| -------------------- | -------------------------------------------- |
| Kubernetes ≥ 1.27    | Cluster with GPU node pool                   |
| KServe ≥ 0.12        | InferenceService CRD                         |
| Knative Serving      | Scale-to-zero autoscaler                     |
| GPU node pool        | NVIDIA L4 / A100 / T4 (≥ 16 GB VRAM)        |
| NVIDIA GPU Operator  | Device plugin + drivers                      |
| HuggingFace token    | For gated models (Llama 3, etc.)             |

---

## 1. Create the Namespace & Secrets

```bash
kubectl apply -f infra/k8s/namespace.yaml

# Edit the secret to add your real tokens BEFORE applying:
#   hf-token         → HuggingFace access token (for gated models)
#   openai-api-key   → only needed if you still use OpenAI as fallback
kubectl apply -f infra/kserve/secrets.yaml
```

---

## 2. Deploy the LLM InferenceService

```bash
kubectl apply -f infra/kserve/llm-inferenceservice.yaml
```

### What this creates

| Resource            | Purpose                                       |
| ------------------- | --------------------------------------------- |
| `InferenceService`  | KServe-managed deployment with Knative pod autoscaler |
| `Secret`            | HuggingFace token for downloading gated model weights |

### Key manifest fields explained

| Field / Annotation                                | Purpose |
| ------------------------------------------------- | ------- |
| `autoscaling.knative.dev/min-scale: "0"`          | **Scale-to-zero** — pods terminate after idle window |
| `autoscaling.knative.dev/max-scale: "2"`          | Caps GPU cost to at most 2 replicas |
| `autoscaling.knative.dev/scale-down-delay: "300s"` | Waits 5 min idle before scaling down (avoids thrashing) |
| `spec.predictor.model.storageUri`                 | HuggingFace model ID (`hf://…`) — **change this to swap models** |
| `spec.predictor.model.runtime: kserve-vllm`       | Uses the vLLM ServingRuntime (OpenAI-compatible API) |
| `nvidia.com/gpu: "1"`                             | Requests 1 GPU; adjust for larger models |
| `--max-model-len=8192`                            | Context window limit; saves VRAM |
| `--gpu-memory-utilization=0.90`                   | vLLM KV-cache budget |
| `--enforce-eager`                                 | Disables CUDA graphs; saves memory on smaller GPUs |

### Wait for readiness

```bash
kubectl -n kubeflow-user get inferenceservice llm-server -w

# NAME         URL                                                   READY
# llm-server   http://llm-server.kubeflow-user.svc.cluster.local     True
```

First startup takes longer because model weights are downloaded from
HuggingFace Hub. Subsequent cold starts are faster if a PVC cache is
configured.

---

## 3. Deploy the RAG Agent

The RAG agent's InferenceService is already configured to call the LLM:

```bash
kubectl apply -f infra/kserve/inference-service.yaml
```

The agent container receives these environment variables:

| Variable          | Value |
| ----------------- | ----- |
| `LLM_BASE_URL`   | `http://llm-server.kubeflow-user.svc.cluster.local/v1` |
| `LLM_MODEL_NAME` | `meta-llama/Meta-Llama-3.1-8B-Instruct` |

The `get_llm()` function in `src/agentic_rag/agent/llm.py` detects
`LLM_BASE_URL` and points `ChatOpenAI(base_url=...)` at the vLLM
endpoint — no code changes needed when swapping models.

---

## 4. Swap the LLM Model

To deploy a different model, edit the `storageUri` and re-apply:

```yaml
# llm-inferenceservice.yaml  →  spec.predictor.model
storageUri: "hf://mistralai/Mistral-7B-Instruct-v0.3"
```

Then update the agent's `LLM_MODEL_NAME` to match:

```yaml
# inference-service.yaml  →  env
- name: LLM_MODEL_NAME
  value: "mistralai/Mistral-7B-Instruct-v0.3"
```

Apply both:

```bash
kubectl apply -f infra/kserve/llm-inferenceservice.yaml
kubectl apply -f infra/kserve/inference-service.yaml
```

### Tested model matrix

| Model                                  | Params | Min GPU   | `storageUri` |
| -------------------------------------- | ------ | --------- | ------------ |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 8 B    | 1 × L4    | `hf://meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `mistralai/Mistral-7B-Instruct-v0.3`    | 7 B    | 1 × L4    | `hf://mistralai/Mistral-7B-Instruct-v0.3` |
| `microsoft/Phi-3-mini-4k-instruct`      | 3.8 B  | 1 × T4    | `hf://microsoft/Phi-3-mini-4k-instruct` |
| `Qwen/Qwen2.5-7B-Instruct`             | 7 B    | 1 × L4    | `hf://Qwen/Qwen2.5-7B-Instruct` |

---

## 5. Verify Scale-to-Zero

```bash
# Watch pods — after 5 min idle the LLM pod should terminate
kubectl -n kubeflow-user get pods -l serving.kserve.io/inferenceservice=llm-server -w

# Send a request to trigger scale-up
curl -X POST \
  http://llm-server.kubeflow-user.svc.cluster.local/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
       "messages": [{"role": "user", "content": "Hello!"}]}'
```

The first request after scale-to-zero triggers a **cold start**
(~60–120 s while the model reloads into GPU memory).

---

## 6. Local Development (without GPU)

For local development you can skip the LLM InferenceService and fall
back to the OpenAI cloud API:

```bash
# .env
OPENAI_API_KEY=sk-...
LLM_MODEL_NAME=gpt-4o-mini
# LLM_BASE_URL left empty → uses OpenAI cloud
```

Or run a lightweight local model with Ollama:

```bash
ollama serve &
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL_NAME=llama3.1
python -m agentic_rag.serving.kserve_runtime
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| Pod stuck in `Pending` | No GPU node available | Check node pool / tolerations / nodeSelector |
| `OOMKilled` | Model too large for allocated memory | Increase `memory` limits or use a smaller `--max-model-len` |
| 504 Gateway Timeout | Cold start too slow | Increase Knative timeout or add PVC model cache |
| `401 Unauthorized` from HF | Missing or invalid HF token | Update `hf-token` in `llm-secrets` |
| Agent returns empty answer | Wrong `LLM_MODEL_NAME` vs `storageUri` | Ensure the model name in the agent env matches the served model |
