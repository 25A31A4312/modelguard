import express from 'express';
import type { Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { getConfiguredGeminiModel, runServerInference } from './geminiService.ts';
import {
  analyzeImagePixels,
  generateAlgorithmicInference,
} from './imageAnalysis.ts';
import { store } from './dbStore.ts';
import { mlflowTracker } from './mlflowService.ts';
import { buildAuditExportRows, generateAuditCSV } from './csvExport.ts';
import type { ImageQualityMetrics, ModelType } from '../src/types/modelguard.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SAMPLES_DIR = path.resolve(__dirname, '../public/samples');

// Enforce safe upload payload ceiling (10 MB decoded binary image max)
export const MAX_IMAGE_UPLOAD_BYTES = 10 * 1024 * 1024;

export const apiRouter = express.Router();

// Static serving for photographic sample assets & persistent uploaded review images
apiRouter.use('/samples', express.static(SAMPLES_DIR));
apiRouter.use('/uploads', express.static(store.uploadsDir));

// 1. Health & Environment Status Check
apiRouter.get('/health', async (_req: Request, res: Response) => {
  const mlflowStatus =
    mlflowTracker.getTrackingUri() !== null
      ? await mlflowTracker.checkConnection()
      : mlflowTracker.getCachedStatus();

  res.json({
    status: 'ok',
    service: 'ModelGuard MLOps Engine',
    uptime: process.uptime(),
    port: Number(process.env.PORT) || 3000,
    hasGeminiApiKey: Boolean(
      process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY'
    ),
    geminiModel: getConfiguredGeminiModel(),
    mlflow: mlflowStatus,
    driftConfig: {
      minSamplesRequired: store.minDriftSamples,
      alpha: store.ksAlpha,
      minRetrainLabeledSamples: store.minRetrainLabeledSamples,
      autoRetrainEnabled: store.autoRetrainEnabled,
    },
    currentObservationsCount: store.inferences.length,
    referenceObservationsCount: store.getActiveReferenceObservations().length,
    timestamp: new Date().toISOString(),
  });
});

// 1b. Persisted Inferences List
apiRouter.get('/inferences', (_req: Request, res: Response) => {
  res.json({
    inferences: store.inferences,
    count: store.inferences.length,
  });
});

// 2. Real-time Inference Endpoint
apiRouter.post('/inference', async (req: Request, res: Response) => {
  try {
    let {
      imageBase64,
      imageUrl,
      mimeType = 'image/jpeg',
      modelType = 'resnet18_classifier',
      enqueueForReview = false,
    } = req.body || {};

    const rawInput = imageBase64 || imageUrl;
    if (!rawInput || typeof rawInput !== 'string') {
      res.status(400).json({ error: 'Missing or invalid image input in request body' });
      return;
    }

    let buffer: Buffer;
    let cleanBase64: string;
    let isSampleAsset = false;
    let sampleAssetUrl = '';

    if (rawInput.startsWith('/api/samples/') || rawInput.startsWith('/samples/')) {
      const filename = path.basename(rawInput.split('?')[0]);
      const filePath = path.join(SAMPLES_DIR, filename);
      if (!fs.existsSync(filePath)) {
        res.status(404).json({ error: `Sample image file '${filename}' not found` });
        return;
      }
      buffer = fs.readFileSync(filePath);
      cleanBase64 = buffer.toString('base64');
      mimeType = filename.endsWith('.png') ? 'image/png' : 'image/jpeg';
      isSampleAsset = true;
      sampleAssetUrl = `/api/samples/${filename}`;
    } else if (rawInput.startsWith('/api/uploads/')) {
      const filename = path.basename(rawInput.split('?')[0]);
      const filePath = path.join(store.uploadsDir, filename);
      if (!fs.existsSync(filePath)) {
        res.status(404).json({ error: `Uploaded image '${filename}' not found` });
        return;
      }
      buffer = fs.readFileSync(filePath);
      cleanBase64 = buffer.toString('base64');
      mimeType = filename.endsWith('.png') ? 'image/png' : 'image/jpeg';
    } else if (rawInput.startsWith('http://') || rawInput.startsWith('https://')) {
      const resp = await fetch(rawInput, { signal: AbortSignal.timeout(8000) });
      if (!resp.ok) {
        res.status(400).json({ error: `Failed to fetch image URL: HTTP ${resp.status}` });
        return;
      }
      const arrayBuf = await resp.arrayBuffer();
      buffer = Buffer.from(arrayBuf);
      cleanBase64 = buffer.toString('base64');
      const ct = resp.headers.get('content-type');
      if (ct) mimeType = ct.split(';')[0].trim();
    } else if (rawInput.startsWith('data:image/svg+xml')) {
      const decodedSvg = rawInput.includes(';base64,')
        ? Buffer.from(rawInput.split(';base64,')[1], 'base64').toString('utf-8')
        : decodeURIComponent(rawInput.split(',')[1] || '');
      buffer = Buffer.from(decodedSvg, 'utf-8');
      cleanBase64 = buffer.toString('base64');
      mimeType = 'image/svg+xml';
    } else {
      const dataUrlMatch = rawInput.match(/^data:(image\/[a-zA-Z0-9+.-]+);base64,([\s\S]+)$/);
      if (dataUrlMatch) {
        mimeType = dataUrlMatch[1];
        cleanBase64 = dataUrlMatch[2];
      } else {
        cleanBase64 = rawInput;
      }
      buffer = Buffer.from(cleanBase64, 'base64');
    }

    if (!buffer || buffer.length < 32) {
      res.status(400).json({ error: 'Corrupt or unsupported image file. Minimum 32 bytes required.' });
      return;
    }

    if (buffer.length > MAX_IMAGE_UPLOAD_BYTES) {
      res.status(413).json({
        error: `Image exceeds maximum safe upload limit of ${Math.round(MAX_IMAGE_UPLOAD_BYTES / (1024 * 1024))} MB.`,
      });
      return;
    }

    // 1. Always compute deterministic pixel-level image quality features from the decoded image buffer
    const pixelQualityMetrics = analyzeImagePixels(buffer);

    // 2. Run multimodal model inference when GEMINI_API_KEY is configured, otherwise deterministic pixel analyzer
    let result = null;
    if (
      process.env.GEMINI_API_KEY &&
      process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY' &&
      !mimeType.includes('svg')
    ) {
      result = await runServerInference(
        cleanBase64,
        mimeType,
        modelType as ModelType,
        pixelQualityMetrics
      );
    }

    if (!result) {
      const activeCkpt = store.getActiveTrainedCheckpoint();
      result = generateAlgorithmicInference(buffer, modelType as ModelType, activeCkpt);
      result.qualityMetrics = pixelQualityMetrics;
    }

    // 3. Persist the complete, un-truncated image to disk and link to result.id
    if (isSampleAsset) {
      result.imageThumbnailUrl = sampleAssetUrl;
    } else {
      result.imageThumbnailUrl = store.persistUploadedImage(result.id, buffer, mimeType);
    }

    // 4. Record observation, route to Human Review if needed, log to MLflow, and evaluate drift auto-triggers
    await store.recordInference(result, { enqueueForReview: Boolean(enqueueForReview) });

    res.json(result);
  } catch (error: any) {
    console.error('[API /inference error]', error);
    res.status(500).json({ error: error.message || 'Internal inference failure' });
  }
});

// 3. Drift Monitoring Metrics (Separate KS Test, Multivariate RBF-MMD, and PSI/Wasserstein)
apiRouter.get('/monitoring/drift', async (req: Request, res: Response) => {
  try {
    const alphaParam = req.query.alpha ? Number(req.query.alpha) : undefined;
    const minSamplesParam = req.query.minSamples ? Number(req.query.minSamples) : undefined;
    const report = await store.evaluateDriftReport({
      alpha: Number.isFinite(alphaParam) ? alphaParam : undefined,
      minSamples: Number.isFinite(minSamplesParam) ? minSamplesParam : undefined,
      checkAutoTrigger: false,
      logToMlflow: false,
    });
    res.json(report);
  } catch (err: any) {
    res.status(500).json({ error: err.message || 'Drift evaluation failed' });
  }
});

// Explicit drift evaluation & trigger check endpoint + threshold configuration update
apiRouter.post('/monitoring/config', async (req: Request, res: Response) => {
  const { minDriftSamples, ksAlpha, minRetrainLabeledSamples, autoRetrainEnabled, activeReferenceDatasetId } =
    req.body || {};

  if (typeof minDriftSamples === 'number' && minDriftSamples >= 2) {
    store.minDriftSamples = Math.floor(minDriftSamples);
  }
  if (typeof ksAlpha === 'number' && ksAlpha > 0 && ksAlpha < 0.5) {
    store.ksAlpha = ksAlpha;
  }
  if (typeof minRetrainLabeledSamples === 'number' && minRetrainLabeledSamples >= 4) {
    store.minRetrainLabeledSamples = Math.floor(minRetrainLabeledSamples);
  }
  if (typeof autoRetrainEnabled === 'boolean') {
    store.autoRetrainEnabled = autoRetrainEnabled;
  }
  if (
    typeof activeReferenceDatasetId === 'string' &&
    store.referenceObservationsByDataset.has(activeReferenceDatasetId)
  ) {
    store.activeReferenceDatasetId = activeReferenceDatasetId;
  }

  store.saveState();
  const report = await store.evaluateDriftReport({ checkAutoTrigger: true, logToMlflow: true });
  res.json(report);
});

// 4. Alerts Center
apiRouter.get('/alerts', (req: Request, res: Response) => {
  const typeFilter = req.query.type as string;
  const statusFilter = req.query.status as string;

  let alerts = store.alerts;
  if (typeFilter && typeFilter !== 'ALL') {
    alerts = alerts.filter((a) => a.type === typeFilter);
  }
  if (statusFilter && statusFilter !== 'ALL') {
    alerts = alerts.filter((a) => a.status === statusFilter);
  }

  res.json({
    alerts,
    summary: {
      total: store.alerts.length,
      open: store.alerts.filter((a) => a.status === 'OPEN').length,
      critical: store.alerts.filter((a) => a.severity === 'CRITICAL' && a.status === 'OPEN').length,
      dataDriftCount: store.alerts.filter((a) => a.type === 'DATA_DRIFT' && a.status === 'OPEN').length,
      lowConfidenceCount: store.alerts.filter(
        (a) => a.type === 'LOW_CONFIDENCE' && a.status === 'OPEN'
      ).length,
    },
  });
});

apiRouter.patch('/alerts/:id', (req: Request, res: Response) => {
  const { id } = req.params;
  const { status } = req.body;

  const alert = store.alerts.find((a) => a.id === id);
  if (!alert) {
    res.status(404).json({ error: 'Alert not found' });
    return;
  }

  if (status) {
    alert.status = status;
    store.saveState();
  }
  res.json(alert);
});

// 5. Human Review Queue
apiRouter.get('/human-review', (_req: Request, res: Response) => {
  res.json({
    queue: store.humanReviewQueue,
    pendingCount: store.humanReviewQueue.filter((q) => q.status === 'PENDING').length,
    approvedForRetraining: store.humanReviewQueue.filter(
      (q) => q.reviewed && q.sendToRetraining && (q.status === 'ACCEPTED' || q.status === 'OVERRIDDEN')
    ).length,
  });
});

// Enqueue a specific inference into Human Review preserving exact inferenceId & persistent image URL
apiRouter.post('/human-review', (req: Request, res: Response) => {
  const { inferenceId, reason } = req.body || {};
  if (!inferenceId) {
    res.status(400).json({ error: 'inferenceId is required' });
    return;
  }

  const inf = store.inferences.find((i) => i.id === inferenceId);
  if (!inf) {
    res.status(404).json({ error: `Inference '${inferenceId}' not found` });
    return;
  }

  const item = store.enqueueHumanReviewFromInference(inf, reason);
  res.status(201).json(item);
});

apiRouter.patch('/human-review/:id', (req: Request, res: Response) => {
  const { id } = req.params;
  const { status, groundTruthLabel, sendToRetraining } = req.body || {};

  const item = store.humanReviewQueue.find((q) => q.id === id);
  if (!item) {
    res.status(404).json({ error: 'Review item not found' });
    return;
  }

  if (status) item.status = status;
  if (typeof groundTruthLabel === 'string') {
    item.groundTruthLabel = groundTruthLabel.trim();
  }
  if (typeof sendToRetraining === 'boolean') {
    item.sendToRetraining = sendToRetraining;
  }

  // Rejected items cannot be sent to retraining
  if (item.status === 'REJECTED') {
    item.sendToRetraining = false;
  }

  item.reviewed = item.status !== 'PENDING';
  item.reviewedAt = new Date().toISOString();
  item.reviewedBy = 'Operator (Human-in-the-Loop)';

  store.syncBaseModelMeasuredMetrics();
  store.saveState();
  res.json(item);
});

// 6. Model Management Registry (with strict Validation Promotion Gate)
apiRouter.get('/models', (_req: Request, res: Response) => {
  res.json({
    models: store.models,
    measuredPerformance: store.getMeasuredPerformanceMetrics(),
  });
});

apiRouter.post('/models/:id/action', (req: Request, res: Response) => {
  const { id } = req.params;
  const { action } = req.body || {}; // 'promote' | 'rollback'

  const targetModel = store.models.find((m) => m.id === id);
  if (!targetModel) {
    res.status(404).json({ error: 'Model not found' });
    return;
  }

  if (action === 'promote') {
    if (targetModel.passedValidation === false) {
      res.status(400).json({
        error: `Promotion blocked by validation gate: Candidate ${targetModel.version} did not pass held-out validation criteria.`,
      });
      return;
    }

    // Archive existing active models of the same family (or learned checkpoints)
    store.models.forEach((m) => {
      if (m.id !== id && m.status === 'ACTIVE' && Boolean(m.checkpointPath) === Boolean(targetModel.checkpointPath)) {
        m.status = 'ARCHIVED';
      }
    });
    targetModel.status = 'ACTIVE';
    targetModel.deployedAt = new Date().toISOString();
  } else if (action === 'rollback') {
    targetModel.status = 'CANDIDATE';
  }

  store.saveState();
  res.json({ success: true, model: targetModel });
});

// 7. Retraining Pipeline (Genuine Supervised Classifier Training + Held-Out Validation)
apiRouter.get('/retrain', (_req: Request, res: Response) => {
  const autoTrigger = store.getAutoTriggerStatus();
  res.json({
    jobs: store.retrainingJobs,
    activeJob:
      store.retrainingJobs.find(
        (j) => j.status === 'TRAINING' || j.status === 'QUEUED' || j.status === 'VALIDATING'
      ) || null,
    eligibility: {
      eligibleLabeledSamplesCount: autoTrigger.eligibleLabeledSamplesCount,
      eligibleDistinctClassesCount: autoTrigger.eligibleDistinctClassesCount,
      minRequiredSamples: store.minRetrainLabeledSamples,
      readyToTrain:
        autoTrigger.eligibleLabeledSamplesCount >= store.minRetrainLabeledSamples &&
        autoTrigger.eligibleDistinctClassesCount >= 2,
    },
    autoTrigger,
  });
});

apiRouter.post('/retrain', async (req: Request, res: Response) => {
  try {
    const { triggerSource = 'MANUAL' } = req.body || {};
    const job = await store.executeRetrainingJob(triggerSource);
    const statusCode = job.status === 'FAILED' ? 422 : 200;
    res.status(statusCode).json(job);
  } catch (err: any) {
    res.status(409).json({ error: err.message || 'Failed to start retraining job' });
  }
});

// 8. Real MLflow Tracking Endpoints (Probes MLFLOW_TRACKING_URI; never fabricates runs)
apiRouter.get('/mlflow/status', async (_req: Request, res: Response) => {
  const status = await mlflowTracker.checkConnection();
  res.json(status);
});

apiRouter.get('/mlflow/experiments', async (_req: Request, res: Response) => {
  const result = await mlflowTracker.listExperiments();
  res.json(result);
});

apiRouter.get('/mlflow/runs', async (req: Request, res: Response) => {
  const expId = req.query.experimentId as string | undefined;
  const result = await mlflowTracker.searchRuns(expId);
  res.json(result);
});

// 9. Full Audit CSV Export Endpoint
apiRouter.get('/export/csv', (_req: Request, res: Response) => {
  try {
    const rows = buildAuditExportRows(store);
    const csvContent = generateAuditCSV(rows);

    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader(
      'Content-Disposition',
      `attachment; filename="modelguard_audit_export_${Date.now()}.csv"`
    );
    res.status(200).send(csvContent);
  } catch (err: any) {
    console.error('[CSV Export Error]', err);
    res.status(500).json({ error: 'Failed to generate audit CSV export' });
  }
});

// 10. Reference Datasets & Reproducible Observation Loader
apiRouter.get('/reference-datasets', (_req: Request, res: Response) => {
  res.json({
    datasets: store.referenceDatasets,
    activeReferenceDatasetId: store.activeReferenceDatasetId,
  });
});

apiRouter.post('/reference-datasets/:id/observations', (req: Request, res: Response) => {
  const { id } = req.params;
  const { observations, imageBase64List, mode = 'replace' } = req.body || {};

  const extracted: ImageQualityMetrics[] = [];
  if (Array.isArray(observations)) {
    for (const o of observations) {
      if (o && typeof o.brightness === 'number' && typeof o.sharpness === 'number') {
        extracted.push({
          brightness: Number(o.brightness),
          contrast: Number(o.contrast ?? 50),
          sharpness: Number(o.sharpness),
          fogProbability: Number(o.fogProbability ?? 0.05),
          edgeDensity: Number(o.edgeDensity ?? 0.2),
          rMean: Number(o.rMean ?? o.brightness),
          gMean: Number(o.gMean ?? o.brightness),
          bMean: Number(o.bMean ?? o.brightness),
          featureVector:
            Array.isArray(o.featureVector) && o.featureVector.length === 8
              ? o.featureVector.map(Number)
              : [
                  Number(o.brightness),
                  Number(o.contrast ?? 50),
                  Number(o.sharpness),
                  Number((Number(o.fogProbability ?? 0.05) * 100).toFixed(2)),
                  Number((Number(o.edgeDensity ?? 0.2) * 100).toFixed(2)),
                  Number(o.rMean ?? o.brightness),
                  Number(o.gMean ?? o.brightness),
                  Number(o.bMean ?? o.brightness),
                ],
          resolution: o.resolution || { width: 640, height: 480 },
        });
      }
    }
  }

  if (Array.isArray(imageBase64List)) {
    for (const raw of imageBase64List) {
      if (typeof raw !== 'string') continue;
      const clean = raw.replace(/^data:image\/[a-zA-Z0-9+.-]+;base64,/, '');
      const buf = Buffer.from(clean, 'base64');
      if (buf.length >= 32) {
        extracted.push(analyzeImagePixels(buf));
      }
    }
  }

  if (extracted.length === 0) {
    res.status(400).json({
      error: 'Provide non-empty observations[] or imageBase64List[] to load reference data.',
    });
    return;
  }

  const updated = store.setReferenceObservations(id, extracted, mode === 'append' ? 'append' : 'replace');
  if (!updated) {
    res.status(404).json({ error: `Reference dataset '${id}' not found` });
    return;
  }

  res.json({ dataset: updated, loadedCount: extracted.length });
});
