# Minimal Kubeflow on kind — Local Setup Guide

> **Goal**: Run KFP pipelines and deploy KServe `InferenceService` resources on a
> local `kind` cluster with **zero cloud dependencies**.

---

## Prerequisites

| Tool | Min Version | Install |
|------|-------------|---------|
| Docker Desktop | 24+ | <https://docs.docker.com/desktop/install/mac-install/> |
| `kind` | 0.23+ | `brew install kind` |
| `kubectl` | 1.28+ | `brew install kubectl` |
| `helm` | 3.14+ | `brew install helm` |

Verify:

```bash
docker version && kind --version && kubectl version --client && helm version --short
```

> **Docker resources**: Allocate at least **6 GB RAM** and **4 CPUs** in Docker
> Desktop → Settings → Resources. The stack is memory-hungry.

---

## 1 — Create the kind Cluster

KServe + Istio need extra port mappings and a non-default API server config.

```bash
cat <<'EOF' > /tmp/kind-kubeflow.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: kubeflow-local
nodes:
  - role: control-plane
    kubeadmConfigPatches:
      - |
        kind: InitConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            node-labels: "ingress-ready=true"
    extraPortMappings:
      # Istio ingress gateway
      - containerPort: 80
        hostPort: 80
        protocol: TCP
      - containerPort: 443
        hostPort: 443
        protocol: TCP
      # KFP UI (NodePort fallback)
      - containerPort: 31080
        hostPort: 31080
        protocol: TCP
EOF

kind create cluster --config /tmp/kind-kubeflow.yaml --wait 120s
```

### Verify

```bash
kubectl cluster-info --context kind-kubeflow-local
kubectl get nodes   # STATUS = Ready
```

### Common failures

| Symptom | Fix |
|---------|-----|
| `ERROR: failed to create cluster` | Ensure Docker is running and has enough resources |
| Node stuck in `NotReady` | Increase Docker memory to 8 GB; delete and recreate cluster |

---

## 2 — Install Istio (required by KServe)

KServe **requires** an ingress/service-mesh layer. Istio minimal profile is the
lightest option.

```bash
# Install istioctl
brew install istioctl

# Install Istio with the minimal profile (no addons)
istioctl install --set profile=minimal --set meshConfig.accessLogFile=/dev/stdout -y

# Wait for Istio pods
kubectl -n istio-system wait --for=condition=Ready pods --all --timeout=300s
```

### Verify

```bash
kubectl get pods -n istio-system
# Expected: istiod-xxx  1/1 Running
```

### Common failures

| Symptom | Fix |
|---------|-----|
| `istiod` CrashLoopBackOff | Not enough cluster memory — increase Docker resources |
| Webhook timeout during later steps | Restart istiod: `kubectl rollout restart deploy/istiod -n istio-system` |

---

## 3 — Install cert-manager (required by KServe)

KServe uses cert-manager for webhook TLS certificates.

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.16.1/cert-manager.yaml

# Wait for cert-manager pods
kubectl -n cert-manager wait --for=condition=Ready pods --all --timeout=180s
```

### Verify

```bash
kubectl get pods -n cert-manager
# Expected: 3 pods (controller, webhook, cainjector) all Running
```

### Common failures

| Symptom | Fix |
|---------|-----|
| Pods pending | Image pull taking time on slow connections — just wait |
| Webhook not ready | `kubectl rollout restart deploy -n cert-manager` |

---

## 4 — Install Kubeflow Pipelines (standalone)

We install the **standalone** KFP deployment (no full Kubeflow platform needed).

> **Note**: Do NOT use `kubectl apply -k "github.com/kubeflow/pipelines/..."` —
> it frequently times out because `kubectl` shells out to `git fetch` on the
> very large pipelines repo. Clone or download the manifests locally instead.

```bash
export KFP_VERSION=2.3.0

# Download and extract the release tarball (faster & more reliable than git)
curl -L "https://github.com/kubeflow/pipelines/archive/refs/tags/${KFP_VERSION}.tar.gz" \
  | tar xz -C /tmp

# Apply cluster-scoped resources from the local copy
kubectl apply -k "/tmp/pipelines-${KFP_VERSION}/manifests/kustomize/cluster-scoped-resources"

# Wait for CRDs to register
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

# Apply platform-agnostic environment
kubectl apply -k "/tmp/pipelines-${KFP_VERSION}/manifests/kustomize/env/platform-agnostic"

# Clean up downloaded manifests
rm -rf "/tmp/pipelines-${KFP_VERSION}"
```

### Wait for all KFP pods (this takes 2-5 min)

```bash
kubectl -n kubeflow wait --for=condition=Ready pods --all --timeout=600s
```

### Verify

```bash
kubectl get pods -n kubeflow
# Expected: ~15 pods all Running/Completed
# Key pods: ml-pipeline-*, mysql-*, minio-*, metadata-grpc-*
```

### Access the KFP UI

```bash
# Port-forward the ML Pipeline UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80 &
```

Open <http://localhost:8080> — you should see the Kubeflow Pipelines dashboard.

### Common failures

| Symptom | Fix |
|---------|-----|
| `ml-pipeline-*` CrashLoopBackOff | Usually MySQL isn't ready yet — delete the pod and let it restart |
| `minio` pod pending | PVC not bound — kind's default StorageClass should work; check `kubectl get pvc -n kubeflow` |
| `metadata-grpc` crash | Depends on MySQL — wait for MySQL first, then restart |

---

## 5 — Install KServe

```bash
# Install KServe CRDs and controller (Serverless mode)
export KSERVE_VERSION=v0.14.1

kubectl apply -f "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve.yaml"

# Wait for KServe controller
kubectl -n kserve wait --for=condition=Ready pods --all --timeout=300s

# Install KServe built-in ClusterServingRuntimes
kubectl apply -f "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve-cluster-resources.yaml"
```

### Verify

```bash
kubectl get pods -n kserve
# Expected: kserve-controller-manager-xxx  Running

kubectl get clusterservingruntimes
# Expected: several runtimes (kserve-lgbserver, kserve-sklearnserver, etc.)
```

### Common failures

| Symptom | Fix |
|---------|-----|
| KServe webhook fails | cert-manager or Istio not ready — verify steps 2 & 3 |
| CRD conflicts | If you previously installed KServe, delete old CRDs first |

---

## 6 — Create the Project Namespace & Deploy ChromaDB

```bash
# Create the namespace used by this project
kubectl apply -f infra/k8s/namespace.yaml

# Label namespace for Istio sidecar injection (needed by KServe)
kubectl label namespace kubeflow-user istio-injection=enabled --overwrite

# Deploy ChromaDB
kubectl apply -f infra/k8s/chroma.yaml

# Wait for ChromaDB
kubectl -n kubeflow-user wait --for=condition=Ready pods -l app=chromadb --timeout=180s
```

### Verify

```bash
kubectl get pods -n kubeflow-user
# Expected: chromadb-xxx  Running

kubectl get svc -n kubeflow-user
# Expected: chroma  ClusterIP  8000
```

---

## 7 — Deploy the KServe InferenceService

```bash
# Create the secret with your API key
# ⚠️  Edit the file first: replace REPLACE_ME with your actual key
kubectl apply -f infra/kserve/secrets.yaml

# Deploy the InferenceService
kubectl apply -f infra/kserve/inference-service.yaml

# Watch rollout (takes 1-3 min)
kubectl -n kubeflow-user get inferenceservice rag-agent --watch
```

Wait until `READY` = `True`:

```
NAME        URL                                           READY
rag-agent   http://rag-agent.kubeflow-user.example.com    True
```

### Test the InferenceService

```bash
# Get the Istio ingress gateway IP (on kind it's localhost)
INGRESS_HOST=localhost
INGRESS_PORT=$(kubectl -n istio-system get svc istio-ingressgateway \
  -o jsonpath='{.spec.ports[?(@.name=="http2")].nodePort}' 2>/dev/null || echo 80)

# Alternatively, port-forward directly:
kubectl port-forward -n kubeflow-user svc/rag-agent-predictor 8081:80 &

curl -X POST http://localhost:8081/v1/models/rag-agent:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"query": "What is agentic RAG?"}]}'
```

### Common failures

| Symptom | Fix |
|---------|-----|
| InferenceService stuck at `Unknown` | Check pod logs: `kubectl logs -n kubeflow-user -l serving.kserve.io/inferenceservice=rag-agent` |
| ImagePullBackOff | Build and load the image into kind first (see below) |
| Revision failed | Check KServe controller logs: `kubectl logs -n kserve -l control-plane=kserve-controller-manager` |

---

## 8 — Build & Load Your Image into kind

kind clusters can't pull from external registries by default. Load images
directly:

```bash
# Build the project image
docker build -t ghcr.io/tilakgupta/agentic-rag-kubeflow:latest .

# Load into the kind cluster
kind load docker-image ghcr.io/tilakgupta/agentic-rag-kubeflow:latest \
  --name kubeflow-local
```

Then set `imagePullPolicy: IfNotPresent` in [infra/kserve/inference-service.yaml](../infra/kserve/inference-service.yaml)
to avoid pulling from the remote registry:

```yaml
spec:
  predictor:
    containers:
      - name: rag-agent
        image: ghcr.io/tilakgupta/agentic-rag-kubeflow:latest
        imagePullPolicy: IfNotPresent   # ← add this
```

---

## 9 — Submit a KFP Pipeline Run

```bash
# Compile the pipeline
python pipelines/rag_pipeline.py --compile

# Port-forward the KFP API (if not already running)
kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888 &

# Submit via the KFP SDK
python -c "
from kfp.client import Client
client = Client(host='http://localhost:8888')
client.create_run_from_pipeline_package(
    'pipelines/compiled/rag_pipeline.yaml',
    arguments={},
    run_name='local-test-run',
)
print('Run submitted — check the UI at http://localhost:8080')
"
```

### Verify

Open <http://localhost:8080> → Runs → you should see `local-test-run`.

---

## Full Stack Verification Checklist

Run this all-in-one check:

```bash
echo "=== Cluster ==="
kubectl get nodes

echo -e "\n=== Istio ==="
kubectl get pods -n istio-system

echo -e "\n=== cert-manager ==="
kubectl get pods -n cert-manager

echo -e "\n=== Kubeflow Pipelines ==="
kubectl get pods -n kubeflow

echo -e "\n=== KServe ==="
kubectl get pods -n kserve
kubectl get clusterservingruntimes

echo -e "\n=== Project namespace ==="
kubectl get pods -n kubeflow-user
kubectl get inferenceservice -n kubeflow-user
```

Everything should show `Running` / `Ready = True`.

---

## Teardown

```bash
# Delete the cluster entirely
kind delete cluster --name kubeflow-local

# Or just remove project resources
kubectl delete -f infra/kserve/inference-service.yaml
kubectl delete -f infra/k8s/chroma.yaml
kubectl delete -f infra/k8s/namespace.yaml
```

---

## Quick Reference — Install Order

```
1. kind cluster
2. Istio  (minimal profile)
3. cert-manager
4. Kubeflow Pipelines  (standalone, platform-agnostic)
5. KServe  (serverless mode)
6. Project namespace + ChromaDB
7. Build & load image → Deploy InferenceService
```

Total install time: **10-15 minutes** on a good connection.
