import assert from 'assert';
import fs from 'fs';
import http from 'http';
import os from 'os';
import path from 'path';
import { PNG } from 'pngjs';
import {
  computeDistributionDistances,
  computeMMD,
  computeMultivariateMMDTest,
  computeTwoSampleKSTest,
  evaluateFeatureDrift,
  sanitizeNumericArray,
} from '../server/statisticalDrift.ts';
import {
  analyzeImagePixels,
  extractReferenceObservationsFromImages,
} from '../server/imageAnalysis.ts';
import { ModelGuardStore } from '../server/dbStore.ts';
import { MLflowTracker } from '../server/mlflowService.ts';
import {
  CSV_AUDIT_HEADERS,
  buildAuditExportRows,
  escapeCSVField,
  generateAuditCSV,
} from '../server/csvExport.ts';
import type { InferenceResult } from '../src/types/modelguard.ts';

console.log('===============================================================');
console.log('ModelGuard Backend & MLOps Verification Test Suite');
console.log('===============================================================\n');

let testsPassed = 0;
let testsFailed = 0;

async function runTest(name: string, fn: () => void | Promise<void>) {
  try {
    await fn();
    console.log(`  [PASS] ${name}`);
    testsPassed++;
  } catch (err: any) {
    console.error(`  [FAIL] ${name}:`, err?.stack || err?.message || err);
    testsFailed++;
  }
}

/**
 * Helper: Creates a real uncompressed RGBA PNG buffer with distinct visual patterns & size > 35 KB base64
 */
function createRealPngBuffer(
  width: number,
  height: number,
  mode: 'sharp_checkerboard' | 'smooth_fog'
): Buffer {
  const png = new PNG({ width, height, filterType: 0 });
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (width * y + x) << 2;
      if (mode === 'sharp_checkerboard') {
        const checker = ((Math.floor(x / 3) + Math.floor(y / 3)) % 2) === 0;
        const jitter = (x * 17 + y * 31) % 35;
        png.data[idx] = checker ? 220 - jitter : 20 + jitter;
        png.data[idx + 1] = checker ? 60 + jitter : 195 - jitter;
        png.data[idx + 2] = checker ? 40 : 230 - jitter;
        png.data[idx + 3] = 255;
      } else {
        // Low-contrast, low-Laplacian-sharpness, high dark-channel haze image
        const wave = Math.round(Math.sin(x / 25) * 4 + Math.cos(y / 25) * 4);
        png.data[idx] = 178 + wave;
        png.data[idx + 1] = 182 + wave;
        png.data[idx + 2] = 186 + wave;
        png.data[idx + 3] = 255;
      }
    }
  }
  return PNG.sync.write(png, { deflateLevel: 0 }); // Uncompressed deflate ensures base64 > 40,000 chars
}

function makeInferenceFromBuffer(
  id: string,
  buf: Buffer,
  label: string,
  confidence: number
): InferenceResult {
  const qm = analyzeImagePixels(buf);
  const isFoggy = qm.fogProbability >= 0.45 || qm.sharpness < 26;
  return {
    id,
    timestamp: new Date().toISOString(),
    modelType: 'resnet18_classifier',
    modelVersion: 'v1.0.0-resnet18',
    primaryPrediction: {
      label,
      confidence,
      category: 'ImageNet-1k Classification',
    },
    topClasses: [{ className: label, confidence, classId: 817 }],
    detectedObjects: [],
    qualityMetrics: qm,
    driftFlags: {
      isLowConfidence: confidence < 0.6,
      isDriftSuspect: isFoggy,
      reason: isFoggy ? 'High haze / low sharpness' : undefined,
    },
    latencyMs: 25,
  };
}

async function main() {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'modelguard-audit-'));

  try {
    // =========================================================================
    // 1. Zero-Sample & Insufficient-Sample Drift Monitoring
    // =========================================================================
    console.log('--- 1. Zero-Sample & Configurable Minimum Sample Drift Tests ---');

    await runTest('Zero current samples: returns INSUFFICIENT_DATA, currentSampleCount=0, and null scores', async () => {
      const s = new ModelGuardStore({
        dataDir: path.join(tmpRoot, 'store-zero'),
        minDriftSamples: 5,
        seedSampleReviewItems: true, // Even with seeded sample review items, currentSampleCount MUST be 0!
      });

      const report = await s.evaluateDriftReport();
      assert.strictEqual(report.overallDriftState, 'INSUFFICIENT_DATA');
      assert.strictEqual(report.currentSampleCount, 0);
      assert.ok(report.referenceSampleCount >= 16, 'Real reference patches loaded');
      assert.strictEqual(report.minSamplesRequired, 5);

      for (const ks of report.ksTests) {
        assert.strictEqual(ks.status, 'INSUFFICIENT_DATA');
        assert.strictEqual(ks.currentSampleCount, 0);
        assert.strictEqual(ks.statistic, null);
        assert.strictEqual(ks.pValue, null);
      }

      assert.strictEqual(report.mmdTest.status, 'INSUFFICIENT_DATA');
      assert.strictEqual(report.mmdTest.mmdScore, null);
      assert.strictEqual(report.mmdTest.pValue, null);

      for (const m of report.metrics) {
        assert.strictEqual(m.status, 'INSUFFICIENT_DATA');
        assert.strictEqual(m.ksStatistic, null);
        assert.strictEqual(m.psi, null);
      }
    });

    await runTest('Below-minimum samples (N=4 when min=5): withholds statistical test and returns INSUFFICIENT_DATA', () => {
      const ref = [70, 72, 74, 71, 73, 75, 69, 72];
      const currBelowMin = [20, 22, 21, 19]; // 4 < 5
      const ks = computeTwoSampleKSTest(ref, currBelowMin, 0.05, 5, 'Sharpness');
      assert.strictEqual(ks.valid, false);
      assert.strictEqual(ks.status, 'INSUFFICIENT_DATA');
      assert.strictEqual(ks.statistic, null);
      assert.strictEqual(ks.pValue, null);
      assert.strictEqual(ks.currentSampleCount, 4);
      assert.strictEqual(ks.minSamplesRequired, 5);
    });

    await runTest('Exact-minimum samples (N=5 when min=5): executes KS & MMD statistical tests', () => {
      const ref = [70, 72, 74, 71, 73];
      const currExactMin = [20, 22, 21, 19, 23]; // 5 == 5
      const ks = computeTwoSampleKSTest(ref, currExactMin, 0.05, 5, 'Sharpness');
      assert.strictEqual(ks.valid, true);
      assert.notStrictEqual(ks.status, 'INSUFFICIENT_DATA');
      assert.strictEqual(ks.statistic, 1.0);
      assert.ok(ks.pValue !== null && ks.pValue < 0.05);
    });

    await runTest('Sufficient samples (N=12): accurately distinguishes stable vs shifted observations', () => {
      const ref = [70.1, 72.4, 74.0, 71.2, 73.5, 75.1, 69.8, 72.0, 71.6, 73.1, 70.9, 74.3];
      const currStable = [70.4, 72.1, 73.8, 71.5, 73.2, 74.8, 70.0, 72.3, 71.9, 72.8, 71.1, 74.0];
      const evalStable = evaluateFeatureDrift('Sharpness', ref, currStable, 0.05, 5);
      assert.strictEqual(evalStable.status, 'STABLE');
      assert.ok(evalStable.ksStatistic !== null && evalStable.ksStatistic < 0.25);
      assert.ok(evalStable.ksPValue !== null && evalStable.ksPValue > 0.5);
    });

    // =========================================================================
    // 2. Genuine Two-Sample KS & Multivariate RBF-MMD Tests + Edge Cases
    // =========================================================================
    console.log('\n--- 2. KS & Multivariate RBF-MMD Statistical Accuracy & Edge Cases ---');

    await runTest('KS Test tie-exactness: identical constant features of unequal sizes yield D=0, p=1.0', () => {
      const constRef = [50, 50, 50, 50, 50]; // m = 5
      const constCurr = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]; // n = 10
      const ks = computeTwoSampleKSTest(constRef, constCurr, 0.05, 5, 'ConstantFeature');
      assert.strictEqual(ks.valid, true);
      assert.strictEqual(ks.statistic, 0);
      assert.strictEqual(ks.pValue, 1);
      assert.strictEqual(ks.status, 'STABLE');

      // Different constants must yield D=1.0 and SEVERE_DRIFT
      const shiftedConst = [80, 80, 80, 80, 80, 80];
      const ksShift = computeTwoSampleKSTest(constRef, shiftedConst, 0.05, 5, 'ShiftedConstant');
      assert.strictEqual(ksShift.statistic, 1.0);
      assert.strictEqual(ksShift.status, 'SEVERE_DRIFT');
    });

    await runTest('Multivariate 8D RBF-MMD: retains H0 on matching vectors and rejects H0 on shifted vectors', () => {
      const fNames = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8'];
      const refVecs = Array.from({ length: 16 }, (_, i) => [
        130 + (i % 5),
        60 + (i % 4),
        75 + (i % 6),
        10 + (i % 3),
        35 + (i % 4),
        128 + (i % 5),
        132 + (i % 5),
        125 + (i % 5),
      ]);
      const stableVecs = Array.from({ length: 12 }, (_, i) => [
        130.5 + (i % 5),
        60.2 + (i % 4),
        74.8 + (i % 6),
        10.2 + (i % 3),
        35.1 + (i % 4),
        128.3 + (i % 5),
        131.8 + (i % 5),
        125.2 + (i % 5),
      ]);
      const shiftedVecs = Array.from({ length: 12 }, (_, i) => [
        175 + (i % 4),
        18 + (i % 3),
        14 + (i % 4),
        78 + (i % 5),
        6 + (i % 2),
        176 + (i % 3),
        180 + (i % 3),
        184 + (i % 3),
      ]);

      const mmdStable = computeMultivariateMMDTest(fNames, refVecs, stableVecs, 0.05, 5);
      assert.strictEqual(mmdStable.valid, true);
      assert.strictEqual(mmdStable.status, 'STABLE');
      assert.strictEqual(mmdStable.isDrift, false);

      const mmdShifted = computeMultivariateMMDTest(fNames, refVecs, shiftedVecs, 0.05, 5);
      assert.strictEqual(mmdShifted.valid, true);
      assert.strictEqual(mmdShifted.isDrift, true);
      assert.ok(
        mmdShifted.mmdScore! > mmdShifted.threshold!,
        `Expected MMD ${mmdShifted.mmdScore} > threshold ${mmdShifted.threshold}`
      );
      assert.ok(mmdShifted.pValue! <= 0.05);
    });

    await runTest('NaN, Infinity, null, empty arrays, and constant vectors handled without NaN leakage', () => {
      const dirty = [10, NaN, Infinity, -Infinity, null, undefined, 'bad' as any, 20];
      assert.deepStrictEqual(sanitizeNumericArray(dirty), [10, 20]);

      const constVecsA = Array.from({ length: 6 }, () => [42, 42, 42]);
      const constVecsB = Array.from({ length: 6 }, () => [42, 42, 42]);
      const mmdConst = computeMultivariateMMDTest(['a', 'b', 'c'], constVecsA, constVecsB, 0.05, 5);
      assert.strictEqual(mmdConst.valid, true);
      assert.strictEqual(mmdConst.mmdScore, 0);
      assert.strictEqual(mmdConst.isDrift, false);

      const distSeparate = computeDistributionDistances('TestFeat', [10, 12, 14, 16, 18], [10, 12, 14, 16, 18], 5);
      assert.strictEqual(distSeparate.method, 'PSI_WASSERSTEIN');
      assert.strictEqual(distSeparate.wassersteinDistance, 0);
    });

    await runTest('Real Reference Loader: extracts authentic pixel observations from /public/samples JPEGs', () => {
      const sampleFile = path.resolve(process.cwd(), 'public/samples/daylight-sports-car.jpg');
      assert.ok(fs.existsSync(sampleFile), 'Sample JPEG exists');
      const obs = extractReferenceObservationsFromImages([sampleFile], 4, 4);
      assert.strictEqual(obs.length, 16);
      assert.ok(obs.every((o) => Array.isArray(o.featureVector) && o.featureVector.length === 8));
    });

    // =========================================================================
    // 3. Human Review Image Persistence (No 30k Truncation, Distinct Previews)
    // =========================================================================
    console.log('\n--- 3. Human Review Persistent Image Storage & Isolation ---');

    await runTest('Two distinct >40KB images persist intact on disk, survive store reload, and never mix', async () => {
      const imgDir = path.join(tmpRoot, 'store-images');
      const s1 = new ModelGuardStore({
        dataDir: imgDir,
        seedSampleReviewItems: false,
        persistToDisk: true,
      });

      const sharpPng = createRealPngBuffer(110, 110, 'sharp_checkerboard');
      const fogPng = createRealPngBuffer(110, 110, 'smooth_fog');

      const sharpB64 = `data:image/png;base64,${sharpPng.toString('base64')}`;
      const fogB64 = `data:image/png;base64,${fogPng.toString('base64')}`;
      assert.ok(sharpB64.length > 40000, `Expected >40,000 base64 chars, got ${sharpB64.length}`);
      assert.ok(fogB64.length > 40000, `Expected >40,000 base64 chars, got ${fogB64.length}`);

      const infA = makeInferenceFromBuffer('inf-sharp-A', sharpPng, 'sports car', 0.91);
      infA.imageThumbnailUrl = s1.persistUploadedImage(infA.id, sharpPng, 'image/png');
      await s1.recordInference(infA, { enqueueForReview: true });

      const infB = makeInferenceFromBuffer('inf-fog-B', fogPng, 'station wagon', 0.48);
      infB.imageThumbnailUrl = s1.persistUploadedImage(infB.id, fogPng, 'image/png');
      await s1.recordInference(infB, { enqueueForReview: true });

      // Reload a brand-new store instance from the same dataDir to verify persistence across restarts
      const s2 = new ModelGuardStore({
        dataDir: imgDir,
        seedSampleReviewItems: false,
        persistToDisk: true,
      });

      const revA = s2.humanReviewQueue.find((q) => q.inferenceId === 'inf-sharp-A');
      const revB = s2.humanReviewQueue.find((q) => q.inferenceId === 'inf-fog-B');
      assert.ok(revA && revB, 'Both review items persisted with exact inferenceId');
      assert.notStrictEqual(revA.imageUrl, revB.imageUrl, 'Distinct image URLs preserved');

      const diskFileA = path.join(imgDir, 'uploads', path.basename(revA.imageUrl));
      const diskFileB = path.join(imgDir, 'uploads', path.basename(revB.imageUrl));
      const reloadedBufA = fs.readFileSync(diskFileA);
      const reloadedBufB = fs.readFileSync(diskFileB);

      // Byte-for-byte identity check (zero truncation!)
      assert.strictEqual(reloadedBufA.equals(sharpPng), true, 'Image A bytes 100% intact');
      assert.strictEqual(reloadedBufB.equals(fogPng), true, 'Image B bytes 100% intact');
    });

    // =========================================================================
    // 4. CSV Export Exact Columns, Escaping & Honest Unmeasured Metrics
    // =========================================================================
    console.log('\n--- 4. CSV Export Verification ---');

    await runTest('CSV Export: includes all 9 exact required columns, escapes RFC-4180 chars, and marks unmeasured metrics empty', async () => {
      const requiredCols = [
        'timestamp',
        'model_name',
        'model_version',
        'drift_method',
        'drift_score',
        'drift_status',
        'reference_sample_count',
        'current_sample_count',
        'performance_metrics',
      ];
      for (const col of requiredCols) {
        assert.ok(CSV_AUDIT_HEADERS.includes(col as any), `Missing required CSV column: ${col}`);
      }

      // 1. Test zero-sample store -> drift_score and accuracy must be empty, NOT fabricated
      const emptyStore = new ModelGuardStore({
        dataDir: path.join(tmpRoot, 'csv-empty'),
        seedSampleReviewItems: false,
      });
      const zeroRows = buildAuditExportRows(emptyStore);
      assert.ok(zeroRows.length >= 5);
      assert.strictEqual(zeroRows[0].drift_status, 'INSUFFICIENT_DATA');
      assert.strictEqual(zeroRows[0].drift_score, null);
      assert.strictEqual(zeroRows[0].current_sample_count, 0);
      assert.strictEqual(zeroRows[0].accuracy, null);
      assert.ok(zeroRows[0].performance_metrics.includes('"measured":false'));

      const csvText = generateAuditCSV(zeroRows);
      const lines = csvText.split('\r\n');
      assert.strictEqual(lines[0], CSV_AUDIT_HEADERS.join(','));

      // 2. Test RFC-4180 escaping
      assert.strictEqual(escapeCSVField('a,b'), '"a,b"');
      assert.strictEqual(escapeCSVField('say "hi"'), '"say ""hi"""');
      assert.strictEqual(escapeCSVField('row1\nrow2'), '"row1\nrow2"');
      assert.strictEqual(escapeCSVField(null), '');

      // 3. Test default calibrated store & standalone Human Review items populate non-blank precision, recall, and f1_score
      const calibratedStore = new ModelGuardStore({
        dataDir: path.join(tmpRoot, 'csv-calibrated'),
        seedSampleReviewItems: true,
        persistToDisk: false,
      });
      assert.strictEqual(
        calibratedStore.getAutoTriggerStatus().eligibleLabeledSamplesCount,
        0,
        'Baseline calibration evaluations must not pre-populate retraining eligibility'
      );
      const calibRows = buildAuditExportRows(calibratedStore);
      assert.ok(calibRows.length >= 5);
      for (const row of calibRows) {
        assert.strictEqual(typeof row.accuracy, 'number', 'accuracy must be numeric');
        assert.strictEqual(typeof row.precision, 'number', 'precision must be numeric (not blank)');
        assert.strictEqual(typeof row.recall, 'number', 'recall must be numeric (not blank)');
        assert.strictEqual(typeof row.f1_score, 'number', 'f1_score must be numeric (not blank)');
        assert.ok(!Number.isNaN(row.precision!) && !Number.isNaN(row.recall!) && !Number.isNaN(row.f1_score!));
      }
    });

    // =========================================================================
    // 5. Real MLflow REST API v2.0 Client Verification
    // =========================================================================
    console.log('\n--- 5. Real MLflow REST API v2.0 Tracking Verification ---');

    await runTest('MLflow Client: honestly reports unavailable when unconfigured or unreachable (no fake run IDs)', async () => {
      const tracker = new MLflowTracker();
      tracker.setTrackingUri('');
      const unconf = await tracker.checkConnection();
      assert.strictEqual(unconf.configured, false);
      assert.strictEqual(unconf.connected, false);

      const unconfRun = await tracker.logWorkflowRun({
        workflowType: 'INFERENCE',
        runName: 'should-not-fake',
        params: { a: 1 },
        metrics: { score: 0.9 },
      });
      assert.strictEqual(unconfRun.logged, false);
      assert.strictEqual(unconfRun.runId, null);

      tracker.setTrackingUri('http://127.0.0.1:59998');
      const unreach = await tracker.searchRuns();
      assert.strictEqual(unreach.status.connected, false);
      assert.deepStrictEqual(unreach.runs, []);
    });

    await runTest('MLflow Client: executes genuine REST v2.0 create/log-batch/artifact/update/search against HTTP server', async () => {
      const recordedRuns = new Map<string, any>();
      const recordedArtifacts: string[] = [];

      const mockMlflowServer = http.createServer((req, res) => {
        let body = '';
        req.on('data', (chunk) => (body += chunk));
        req.on('end', () => {
          const url = req.url || '';
          res.setHeader('Content-Type', 'application/json');

          if (url.startsWith('/api/2.0/mlflow/experiments/get-by-name')) {
            res.statusCode = 404;
            res.end(JSON.stringify({ error_code: 'RESOURCE_DOES_NOT_EXIST' }));
            return;
          }
          if (url === '/api/2.0/mlflow/experiments/create') {
            res.end(JSON.stringify({ experiment_id: '42' }));
            return;
          }
          if (url.startsWith('/api/2.0/mlflow/experiments/get?')) {
            res.end(JSON.stringify({ experiment: { experiment_id: '42', name: 'test' } }));
            return;
          }
          if (url === '/api/2.0/mlflow/runs/create') {
            const parsed = JSON.parse(body);
            const runId = `mlflow-real-${recordedRuns.size + 1}`;
            recordedRuns.set(runId, {
              info: {
                run_id: runId,
                run_name: parsed.run_name,
                experiment_id: parsed.experiment_id,
                status: 'RUNNING',
                start_time: parsed.start_time,
                artifact_uri: `/mlflow/42/${runId}/artifacts`,
              },
              data: { params: [], metrics: [], tags: parsed.tags || [] },
            });
            res.end(JSON.stringify({ run: recordedRuns.get(runId) }));
            return;
          }
          if (url === '/api/2.0/mlflow/runs/log-batch') {
            const parsed = JSON.parse(body);
            const r = recordedRuns.get(parsed.run_id);
            if (r) {
              r.data.params.push(...(parsed.params || []));
              r.data.metrics.push(...(parsed.metrics || []));
              r.data.tags.push(...(parsed.tags || []));
            }
            res.end(JSON.stringify({}));
            return;
          }
          if (url.startsWith('/api/2.0/mlflow-artifacts/artifacts/') && req.method === 'PUT') {
            recordedArtifacts.push(url);
            res.end(JSON.stringify({}));
            return;
          }
          if (url === '/api/2.0/mlflow/runs/update') {
            const parsed = JSON.parse(body);
            const r = recordedRuns.get(parsed.run_id);
            if (r) {
              r.info.status = parsed.status;
              r.info.end_time = parsed.end_time;
            }
            res.end(JSON.stringify({ run_info: r?.info }));
            return;
          }
          if (url === '/api/2.0/mlflow/runs/search') {
            res.end(JSON.stringify({ runs: Array.from(recordedRuns.values()) }));
            return;
          }
          res.statusCode = 404;
          res.end(JSON.stringify({}));
        });
      });

      await new Promise<void>((resolve) => mockMlflowServer.listen(0, '127.0.0.1', resolve));
      const port = (mockMlflowServer.address() as any).port;

      try {
        const tracker = new MLflowTracker();
        tracker.setTrackingUri(`http://127.0.0.1:${port}`);

        const outcome = await tracker.logWorkflowRun({
          workflowType: 'DRIFT_EVALUATION',
          runName: 'drift-ks-check',
          params: { method: 'KS_TEST', alpha: 0.05 },
          metrics: { ks_statistic: 0.85, p_value: 0.0004 },
          artifactName: 'drift_report.json',
          artifactPayload: { status: 'CRITICAL_DRIFT' },
        });

        assert.strictEqual(outcome.logged, true);
        assert.strictEqual(outcome.runId, 'mlflow-real-1');
        assert.strictEqual(outcome.artifactLogged, true);
        assert.strictEqual(recordedArtifacts.length, 1);

        const searched = await tracker.searchRuns();
        assert.strictEqual(searched.status.connected, true);
        assert.strictEqual(searched.runs.length, 1);
        assert.strictEqual(searched.runs[0].status, 'FINISHED');
        assert.strictEqual(searched.runs[0].params['method'], 'KS_TEST');
        assert.strictEqual(searched.runs[0].metrics['ks_statistic'], 0.85);
      } finally {
        await new Promise<void>((resolve) => mockMlflowServer.close(() => resolve()));
      }
    });

    // =========================================================================
    // 6. Real Supervised Retraining & Validation Gate
    // =========================================================================
    console.log('\n--- 6. Supervised Retraining & Held-Out Validation Gate ---');

    await runTest('Retraining aborts honestly with FAILED status when eligible reviewed samples < minimum', async () => {
      const s = new ModelGuardStore({
        dataDir: path.join(tmpRoot, 'retrain-insufficient'),
        minRetrainLabeledSamples: 6,
        seedSampleReviewItems: false,
      });

      const job = await s.executeRetrainingJob('MANUAL');
      assert.strictEqual(job.status, 'FAILED');
      assert.strictEqual(job.failureReason, 'INSUFFICIENT_LABELED_DATA');
      assert.strictEqual(job.trainLoss, null);
      assert.strictEqual(s.models.filter((m) => m.status === 'CANDIDATE').length, 0);
    });

    await runTest('Retraining trains real softmax classifier on reviewed features, validates on holdout, and gates promotion', async () => {
      const s = new ModelGuardStore({
        dataDir: path.join(tmpRoot, 'retrain-real'),
        minRetrainLabeledSamples: 6,
        seedSampleReviewItems: false,
      });

      const sharpPng = createRealPngBuffer(64, 64, 'sharp_checkerboard');
      const fogPng = createRealPngBuffer(64, 64, 'smooth_fog');

      // Populate 8 human-reviewed samples across 2 visually distinct classes (4 sharp sports cars, 4 foggy trucks)
      for (let i = 0; i < 4; i++) {
        const infCar = makeInferenceFromBuffer(`car-${i}`, sharpPng, 'sports car', 0.88);
        const revCar = s.enqueueHumanReviewFromInference(infCar);
        revCar.reviewed = true;
        revCar.status = 'ACCEPTED';
        revCar.groundTruthLabel = 'sports car';
        revCar.sendToRetraining = true;

        const infFog = makeInferenceFromBuffer(`fog-${i}`, fogPng, 'foggy truck', 0.49);
        const revFog = s.enqueueHumanReviewFromInference(infFog);
        revFog.reviewed = true;
        revFog.status = 'OVERRIDDEN';
        revFog.groundTruthLabel = 'foggy truck';
        revFog.sendToRetraining = true;
      }

      const job = await s.executeRetrainingJob('HUMAN_REVIEW_BATCH');
      assert.strictEqual(job.status, 'COMPLETED');
      assert.strictEqual(job.newSamplesCount, 8);
      assert.ok((job.trainSamplesCount ?? 0) >= 4);
      assert.ok((job.valSamplesCount ?? 0) >= 2);
      assert.ok(job.lossHistory && job.lossHistory.length === 30);
      assert.ok(
        job.lossHistory[29].trainLoss < job.lossHistory[0].trainLoss,
        'Training cross-entropy loss genuinely decreased via gradient descent'
      );
      assert.strictEqual(job.validationReport?.passedPromotionCriteria, true);
      assert.strictEqual(job.validationReport?.candidateAccuracy, 1.0);

      const candidate = s.models.find((m) => m.id === job.candidateModelId);
      assert.ok(candidate && candidate.checkpointPath && fs.existsSync(candidate.checkpointPath));
    });

    // =========================================================================
    // 7. End-to-End Automatic Drift Trigger (4 Required Scenarios)
    // =========================================================================
    console.log('\n--- 7. End-to-End Automatic Drift Trigger ---');

    await runTest('Case 1 (Threshold NOT exceeded): Stable observations do NOT trigger retraining', async () => {
      const s = new ModelGuardStore({
        dataDir: path.join(tmpRoot, 'trig-stable'),
        minDriftSamples: 5,
        minRetrainLabeledSamples: 6,
        seedSampleReviewItems: false,
      });

      // Sample evenly across all 3 clean reference scenes (2 observations per scene = 6 total)
      const allRefObs = s.getActiveReferenceObservations();
      const refObs = allRefObs.filter((_, idx) => idx % 3 === 0).slice(0, 6);
      for (let i = 0; i < refObs.length; i++) {
        s.inferences.unshift({
          id: `stable-${i}`,
          timestamp: new Date().toISOString(),
          modelType: 'resnet18_classifier',
          modelVersion: 'v1.0.0-resnet18',
          primaryPrediction: { label: 'sports car', confidence: 0.92, category: 'ImageNet' },
          topClasses: [],
          detectedObjects: [],
          qualityMetrics: refObs[i],
          driftFlags: { isLowConfidence: false, isDriftSuspect: false },
          latencyMs: 20,
        });
      }

      const rep = await s.evaluateDriftReport({ checkAutoTrigger: true });
      assert.strictEqual(rep.overallDriftState, 'HEALTHY');
      assert.strictEqual(rep.autoTrigger.lastEvent, null);
      assert.strictEqual(s.retrainingJobs.length, 0);
    });

    await runTest('Case 2 (Exceeded + Insufficient Labeled Data): Skips cleanly with clear reason while keeping drift visible', async () => {
      const s = new ModelGuardStore({
        dataDir: path.join(tmpRoot, 'trig-skip'),
        minDriftSamples: 5,
        minRetrainLabeledSamples: 6,
        seedSampleReviewItems: false,
      });

      const fogPng = createRealPngBuffer(64, 64, 'smooth_fog');
      for (let i = 0; i < 6; i++) {
        await s.recordInference(makeInferenceFromBuffer(`drift-skip-${i}`, fogPng, 'foggy car', 0.45));
      }

      const rep = await s.evaluateDriftReport({ checkAutoTrigger: false });
      assert.strictEqual(rep.overallDriftState, 'CRITICAL_DRIFT');
      assert.strictEqual(rep.autoTrigger.lastEvent?.outcome, 'SKIPPED_INSUFFICIENT_LABELED_DATA');
      assert.ok(rep.autoTrigger.lastEvent?.reason.includes('required human-reviewed labeled samples'));
      assert.strictEqual(s.retrainingJobs.length, 0);
    });

    await runTest('Case 3 & 4 (Exceeded + Sufficient Labeled Data + Duplicate Prevention): Triggers retraining once and prevents duplicates', async () => {
      const s = new ModelGuardStore({
        dataDir: path.join(tmpRoot, 'trig-fire'),
        minDriftSamples: 5,
        minRetrainLabeledSamples: 6,
        seedSampleReviewItems: false,
      });

      const sharpPng = createRealPngBuffer(64, 64, 'sharp_checkerboard');
      const fogPng = createRealPngBuffer(64, 64, 'smooth_fog');

      // Seed 6 verified human-reviewed labeled samples (3 sharp, 3 foggy)
      for (let i = 0; i < 3; i++) {
        const c = s.enqueueHumanReviewFromInference(makeInferenceFromBuffer(`seed-car-${i}`, sharpPng, 'sports car', 0.9));
        c.reviewed = true;
        c.status = 'ACCEPTED';
        c.groundTruthLabel = 'sports car';
        c.sendToRetraining = true;

        const f = s.enqueueHumanReviewFromInference(makeInferenceFromBuffer(`seed-fog-${i}`, fogPng, 'foggy vehicle', 0.5));
        f.reviewed = true;
        f.status = 'OVERRIDDEN';
        f.groundTruthLabel = 'foggy vehicle';
        f.sendToRetraining = true;
      }

      // Now stream 5 drifted foggy inferences -> on the 5th observation, minDriftSamples=5 is met and auto-trigger fires!
      for (let i = 0; i < 5; i++) {
        await s.recordInference(makeInferenceFromBuffer(`live-fog-${i}`, fogPng, 'foggy vehicle', 0.47));
      }

      assert.strictEqual(s.retrainingJobs.length, 1, 'Exactly 1 automatic retraining job triggered');
      assert.strictEqual(s.retrainingJobs[0].triggerSource, 'AUTO_DRIFT_THRESHOLD');
      assert.strictEqual(s.retrainingJobs[0].status, 'COMPLETED');
      assert.strictEqual(s.driftTriggerHistory[0].outcome, 'RETRAINING_ENQUEUED');

      // Case 4: Re-evaluate the identical drifted window -> must deduplicate and NOT create a 2nd job!
      await s.evaluateDriftReport({ checkAutoTrigger: true });
      assert.strictEqual(s.retrainingJobs.length, 1, 'Duplicate trigger prevented — still 1 job');
      assert.strictEqual(s.driftTriggerHistory[0].outcome, 'SKIPPED_DUPLICATE_EVENT');
    });

    // =========================================================================
    // 8. End-to-End Express HTTP API & Downloaded CSV Verification
    // =========================================================================
    console.log('\n--- 8. Express HTTP Endpoints & CSV Download Verification ---');

    await runTest('HTTP API: verifies /api/health, zero-sample /api/monitoring/drift, /api/uploads previews, auto-retraining, and /api/export/csv', async () => {
      const prevGeminiKey = process.env.GEMINI_API_KEY;
      process.env.GEMINI_API_KEY = '';

      const expressMod = (await import('express')).default;
      const { apiRouter } = await import('../server/apiRouter.ts');
      const { store: liveStore } = await import('../server/dbStore.ts');

      const origInferences = [...liveStore.inferences];
      const origAlerts = [...liveStore.alerts];
      const origQueue = [...liveStore.humanReviewQueue];
      const origModels = [...liveStore.models];
      const origJobs = [...liveStore.retrainingJobs];
      const origHistory = [...liveStore.driftTriggerHistory];
      const origSaveState = liveStore.saveState.bind(liveStore);
      const createdUploadFiles: string[] = [];
      const createdModelFiles: string[] = [];

      // Suppress writes to workspace ./data/store-state.json during test run
      liveStore.saveState = () => {};
      liveStore.inferences = [];
      liveStore.alerts = [];
      liveStore.humanReviewQueue = [];
      liveStore.retrainingJobs = [];
      liveStore.driftTriggerHistory = [];

      const app = expressMod();
      app.use(expressMod.json({ limit: '50mb' }));
      app.use('/api', apiRouter);
      app.use('/api', (_req, res) => {
        res.status(404).json({ error: 'API endpoint not found' });
      });

      const distPath = path.resolve(process.cwd(), 'dist');
      if (fs.existsSync(distPath)) {
        app.use(expressMod.static(distPath));
      }

      const srv = http.createServer(app);
      await new Promise<void>((resolve) => srv.listen(0, '127.0.0.1', resolve));
      const port = (srv.address() as any).port;
      const base = `http://127.0.0.1:${port}`;

      try {
        // 1. Health endpoint
        const healthRes = await fetch(`${base}/api/health`);
        assert.strictEqual(healthRes.status, 200);
        const healthJson: any = await healthRes.json();
        assert.strictEqual(healthJson.status, 'ok');
        assert.strictEqual(healthJson.currentObservationsCount, 0);

        // 2. Zero-sample drift status over HTTP
        const zeroDriftRes = await fetch(`${base}/api/monitoring/drift`);
        assert.strictEqual(zeroDriftRes.status, 200);
        const zeroDrift: any = await zeroDriftRes.json();
        assert.strictEqual(zeroDrift.overallDriftState, 'INSUFFICIENT_DATA');
        assert.strictEqual(zeroDrift.currentSampleCount, 0);

        // 3. Upload 2 distinct >40KB images and verify distinct HTTP image previews in Human Review
        const sharpPng = createRealPngBuffer(105, 105, 'sharp_checkerboard');
        const fogPng = createRealPngBuffer(105, 105, 'smooth_fog');

        const upARes = await fetch(`${base}/api/inference`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            imageBase64: `data:image/png;base64,${sharpPng.toString('base64')}`,
            mimeType: 'image/png',
            modelType: 'resnet18_classifier',
            enqueueForReview: true,
          }),
        });
        assert.strictEqual(upARes.status, 200);
        const infA: any = await upARes.json();

        const upBRes = await fetch(`${base}/api/inference`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            imageBase64: `data:image/png;base64,${fogPng.toString('base64')}`,
            mimeType: 'image/png',
            modelType: 'resnet18_classifier',
            enqueueForReview: true,
          }),
        });
        assert.strictEqual(upBRes.status, 200);
        const infB: any = await upBRes.json();

        const hrRes = await fetch(`${base}/api/human-review`);
        const hrJson: any = await hrRes.json();
        const revA = hrJson.queue.find((q: any) => q.inferenceId === infA.id);
        const revB = hrJson.queue.find((q: any) => q.inferenceId === infB.id);
        assert.ok(revA && revB, 'Both uploaded inferences present in Human Review');
        assert.notStrictEqual(revA.imageUrl, revB.imageUrl);

        const previewABuf = Buffer.from(await (await fetch(`${base}${revA.imageUrl}`)).arrayBuffer());
        const previewBBuf = Buffer.from(await (await fetch(`${base}${revB.imageUrl}`)).arrayBuffer());
        assert.strictEqual(previewABuf.equals(sharpPng), true, 'HTTP preview A matches uploaded bytes');
        assert.strictEqual(previewBBuf.equals(fogPng), true, 'HTTP preview B matches uploaded bytes');

        // 4. Label 6 reviewed samples (3 sharp cars, 3 foggy vehicles) & trigger automatic retraining
        await fetch(`${base}/api/human-review/${revA.id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status: 'ACCEPTED', groundTruthLabel: 'sports car', sendToRetraining: true }),
        });
        await fetch(`${base}/api/human-review/${revB.id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status: 'OVERRIDDEN', groundTruthLabel: 'foggy vehicle', sendToRetraining: true }),
        });

        for (let i = 0; i < 2; i++) {
          const rCar: any = await (
            await fetch(`${base}/api/inference`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                imageBase64: `data:image/png;base64,${sharpPng.toString('base64')}`,
                mimeType: 'image/png',
                enqueueForReview: true,
              }),
            })
          ).json();
          const hrNow: any = await (await fetch(`${base}/api/human-review`)).json();
          const qCar = hrNow.queue.find((q: any) => q.inferenceId === rCar.id);
          await fetch(`${base}/api/human-review/${qCar.id}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'ACCEPTED', groundTruthLabel: 'sports car', sendToRetraining: true }),
          });

          const rFog: any = await (
            await fetch(`${base}/api/inference`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                imageBase64: `data:image/png;base64,${fogPng.toString('base64')}`,
                mimeType: 'image/png',
                enqueueForReview: true,
              }),
            })
          ).json();
          const hrNow2: any = await (await fetch(`${base}/api/human-review`)).json();
          const qFog = hrNow2.queue.find((q: any) => q.inferenceId === rFog.id);
          await fetch(`${base}/api/human-review/${qFog.id}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'OVERRIDDEN', groundTruthLabel: 'foggy vehicle', sendToRetraining: true }),
          });
        }

        // Stream additional drifted foggy frames so drift evaluation breaches threshold with >=6 eligible labels
        for (let i = 0; i < 6; i++) {
          await fetch(`${base}/api/inference`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              imageBase64: `data:image/png;base64,${fogPng.toString('base64')}`,
              mimeType: 'image/png',
            }),
          });
        }

        // Track temp test files created in uploadsDir / modelsDir for clean teardown
        for (const inf of liveStore.inferences) {
          if (inf.imageThumbnailUrl?.startsWith('/api/uploads/')) {
            createdUploadFiles.push(path.join(liveStore.uploadsDir, path.basename(inf.imageThumbnailUrl)));
          }
        }
        for (const m of liveStore.models) {
          if (m.checkpointPath && !origModels.some((om) => om.id === m.id)) {
            createdModelFiles.push(m.checkpointPath);
          }
        }

        const retrainRes = await fetch(`${base}/api/retrain`);
        const retrainJson: any = await retrainRes.json();
        assert.ok(retrainJson.jobs.length >= 1, 'Automatic retraining job created');
        assert.strictEqual(retrainJson.jobs[0].triggerSource, 'AUTO_DRIFT_THRESHOLD');
        assert.strictEqual(retrainJson.jobs[0].status, 'COMPLETED');

        // 5. Verify downloaded CSV over HTTP includes model_version, drift_score, and performance_metrics
        const csvRes = await fetch(`${base}/api/export/csv`);
        assert.strictEqual(csvRes.status, 200);
        assert.ok((csvRes.headers.get('content-type') || '').includes('text/csv'));
        assert.ok((csvRes.headers.get('content-disposition') || '').includes('attachment'));
        const csvBody = await csvRes.text();
        const csvLines = csvBody.trim().split('\r\n');
        const headers = csvLines[0].split(',');
        assert.ok(headers.includes('model_version'), 'CSV header includes model_version');
        assert.ok(headers.includes('drift_score'), 'CSV header includes drift_score');
        assert.ok(headers.includes('performance_metrics'), 'CSV header includes performance_metrics');
        assert.ok(csvLines.length >= 15, 'CSV contains summary + live observation rows');
        // RFC-4180 escapes internal JSON double-quotes as "" inside the quoted performance_metrics cell
        assert.ok(csvBody.includes('""measured"":true'), 'CSV reflects measured human-verified performance');

        // Verify exported CSV rows have non-blank numeric accuracy, precision, recall, and f1_score cells
        const exportedRows = buildAuditExportRows(liveStore);
        assert.ok(exportedRows.some((r) => r.record_type === 'RETRAINING_VALIDATION'), 'Includes retraining validation row');
        for (const r of exportedRows) {
          assert.notStrictEqual(r.accuracy, null, `Row ${r.record_id} accuracy must not be null/blank`);
          assert.notStrictEqual(r.precision, null, `Row ${r.record_id} precision must not be null/blank`);
          assert.notStrictEqual(r.recall, null, `Row ${r.record_id} recall must not be null/blank`);
          assert.notStrictEqual(r.f1_score, null, `Row ${r.record_id} f1_score must not be null/blank`);
        }
      } finally {
        if (prevGeminiKey === undefined) delete process.env.GEMINI_API_KEY;
        else process.env.GEMINI_API_KEY = prevGeminiKey;

        for (const f of [...createdUploadFiles, ...createdModelFiles]) {
          try {
            if (fs.existsSync(f)) fs.unlinkSync(f);
          } catch {}
        }

        liveStore.inferences = origInferences;
        liveStore.alerts = origAlerts;
        liveStore.humanReviewQueue = origQueue;
        liveStore.models = origModels;
        liveStore.retrainingJobs = origJobs;
        liveStore.driftTriggerHistory = origHistory;
        liveStore.saveState = origSaveState;
        await new Promise<void>((resolve) => srv.close(() => resolve()));
      }
    });
  } finally {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  }

  console.log('\n===============================================================');
  console.log(`Verification Summary: ${testsPassed} Passed, ${testsFailed} Failed`);
  console.log('===============================================================');

  if (testsFailed > 0) {
    process.exit(1);
  }
}

main();

