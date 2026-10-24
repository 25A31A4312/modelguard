# ModelGuard — Computer Vision Drift Monitoring & MLOps Platform

ModelGuard is a full-stack computer vision monitoring and MLOps platform built with React, TypeScript, Vite, and Express. It provides individual image inference, two-sample Kolmogorov–Smirnov (KS) and Multivariate Gaussian RBF-Kernel Maximum Mean Discrepancy (MMD) drift detection, persistent Human-in-the-Loop review, MLflow REST API v2.0 tracking, validation-gated supervised retraining, RFC-4180 audit CSV export, and Docker Compose orchestration.

---

## Core Architecture & Engineering Guarantees

### 1. Real Pixel Feature Extraction & Reference Baselines (`server/imageAnalysis.ts`)
- Incoming JPEG and PNG images are decoded into uncompressed RGBA pixel matrices (`jpeg-js` and `pngjs`).
- Deterministic 8D feature vectors are computed directly from decoded pixels:
  1. `brightness` — ITU-R BT.601 luminance mean (`[0, 255]`)
  2. `contrast` — normalized RMS luminance standard deviation (`[0, 100]`)
  3. `sharpness` — 2D discrete Laplacian variance score (`[0, 100]`)
  4. `fog_index` — He et al. Dark Channel Prior + contrast/edge attenuation (`[0, 100]`)
  5. `edge_density` — 2D Sobel gradient magnitude ratio (`[0, 100]`)
  6. `red_mean`, `green_mean`, `blue_mean` — RGB channel means (`[0, 255]`)
- Reference datasets are extracted reproducibly from real photographic images in `public/samples/` using non-overlapping $4 \times 4$ spatial sub-patches (32 real observations per baseline). Custom reference observations or images can also be loaded via `POST /api/reference-datasets/:id/observations`.

### 2. Statistical Drift Engine (`server/statisticalDrift.ts`)
- **Zero / Insufficient Sample Safety**: Configurable `DRIFT_MIN_SAMPLES` (default `5`) is enforced for both reference ($m$) and current ($n$) observations. When $m < \text{min}$ or $n < \text{min}$ (including $n = 0$), no statistical test is run; status is `INSUFFICIENT_DATA` and all test statistics/scores are `null`.
- **Two-Sample Kolmogorov–Smirnov (KS) Test (1D)**: Tie-exact ECDF supremum $D_{m,n} = \sup_t |F_m(t) - G_n(t)|$ with Stephens (1970) finite-sample asymptotic survival $p$-value ($\alpha = 0.05$ default).
- **Multivariate Maximum Mean Discrepancy (MMD) Test (8D Vectors)**: Gaussian RBF kernel $k(\mathbf{x}, \mathbf{y}) = \exp(-\|\mathbf{x} - \mathbf{y}\|_2^2 / (2\sigma^2))$ on reference-standardized coordinates using the pairwise median heuristic for $\sigma$ and a deterministic $B = 200$ Gram-matrix permutation test for empirical $p$-value and $(1 - \alpha)$ critical threshold.
- **Descriptive Distances**: `PSI` (10-bin smoothed) and `Wasserstein-1` are computed and displayed separately from KS and MMD hypothesis tests.

### 3. Persistent Human Review Queue (`server/dbStore.ts`, `server/apiRouter.ts`)
- Uploaded binary images (validated up to `10 MB` max) are written intact to `${MODELGUARD_DATA_DIR}/uploads/<inferenceId>.<ext>` without truncation and served via `/api/uploads/:filename`.
- Review items retain exact `inferenceId` associations and survive navigation and server restarts via `${MODELGUARD_DATA_DIR}/store-state.json`.

### 4. Audit CSV Export (`GET /api/export/csv`)
- Exports RFC-4180 compliant CSV containing required columns:
  `timestamp`, `model_name`, `model_version`, `drift_method`, `drift_score`, `drift_status`, `reference_sample_count`, `current_sample_count`, `performance_metrics`
  alongside `ks_statistic`, `ks_p_value`, `mmd_score`, `mmd_p_value`, `accuracy`, `precision`, `recall`, `f1_score`, `inference_latency_ms`, `predicted_class`, `ground_truth_label`, and `review_status`.
- Unmeasured ground-truth metrics (when 0 inferences have human-verified labels) are exported as empty cells (`""`) with `"measured":false` documented inside `performance_metrics`.

### 5. Real MLflow Tracking (`server/mlflowService.ts`)
- Speaks directly to an MLflow Tracking Server via REST API v2.0 (`MLFLOW_TRACKING_URI`).
- Logs experiments, runs, parameters, metrics, and JSON artifacts for inference, drift evaluations, drift triggers, and retraining jobs.
- When `MLFLOW_TRACKING_URI` is unset or unreachable, the API and UI report `UNCONFIGURED` / `UNREACHABLE` honestly with zero fabricated runs.

### 6. Supervised Retraining, Validation Gate & Automatic Drift Trigger (`server/retrainingEngine.ts`)
- Uses strictly human-reviewed samples (`ACCEPTED` / `OVERRIDDEN`, `sendToRetraining: true`) with valid ground-truth labels across $\ge 2$ classes and $\ge$ `RETRAIN_MIN_LABELED_SAMPLES` (default `6`).
- Trains an L2-regularized Multinomial Softmax Logistic Regression classifier on standardized training split features and validates on a deterministic stratified held-out split.
- Candidate models that fail held-out validation criteria are blocked from production promotion (`HTTP 400`).
- When a live drift evaluation exceeds threshold, ModelGuard automatically enqueues retraining if eligible labeled data meets minimum requirements (with duplicate-event deduplication), or records `SKIPPED_INSUFFICIENT_LABELED_DATA` while preserving full drift telemetry.

---

## Local Development & Verification

```bash
# 1. Install dependencies
npm install

# 2. Configure environment variables
cp .env.example .env

# 3. Run TypeScript static verification & automated test suite
npm run lint
npm test

# 4. Build production frontend & start server on port 3000
npm run build
npm start
```

---

## Docker & Docker Compose (App + Official MLflow Tracking Server)

- **`Dockerfile`**: Multi-stage `node:22-bookworm-slim` image that installs dependencies reproducibly from `bun.lock` (`bun install --frozen-lockfile`), builds the Vite SPA into `/app/dist`, installs production-only runtime dependencies (`--frozen-lockfile --production`), persists state under `MODELGUARD_DATA_DIR=/app/data`, runs as non-root `node`, and starts Express directly via `node --experimental-strip-types server.ts` (`NODE_ENV=production`).
- **`docker-compose.yml`**: Orchestrates both the official MLflow Tracking Server (`ghcr.io/mlflow/mlflow:v2.16.2` on port `5000` with SQLite backend + `--artifacts-destination /mlflow/artifacts --serve-artifacts`) and the production `modelguard` service (port `3000`) on a shared bridge network (`modelguard_net`), defaulting `MLFLOW_TRACKING_URI` to `http://mlflow:5000`.

```bash
# Build and start ModelGuard + MLflow Tracking Server
docker compose up --build -d

# Check container health & MLflow connectivity
curl -s http://localhost:3000/api/health
curl -s http://localhost:5000/health

# View logs
docker compose logs -f modelguard

# Stop stack cleanly
docker compose down
```

