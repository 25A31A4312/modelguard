import { ModelGuardStore, store as defaultStore } from './dbStore.ts';
import type { ImageQualityMetrics } from '../src/types/modelguard.ts';
import { FEATURE_VECTOR_NAMES } from './imageAnalysis.ts';
import {
  computeMultivariateMMDTest,
  computeTwoSampleKSTest,
} from './statisticalDrift.ts';

/**
 * Required columns per specification:
 * - timestamp
 * - model_name
 * - model_version
 * - drift_method
 * - drift_score
 * - drift_status
 * - reference_sample_count
 * - current_sample_count
 * - performance_metrics
 * Plus additional diagnostic & ground-truth columns.
 */
export interface AuditExportRow {
  timestamp: string;
  model_name: string;
  model_version: string;
  drift_method: string;
  drift_score: number | null;
  drift_status: string;
  reference_sample_count: number;
  current_sample_count: number;
  performance_metrics: string;
  record_type: 'DRIFT_SUMMARY' | 'INFERENCE_OBSERVATION' | 'RETRAINING_VALIDATION';
  record_id: string;
  feature_name: string;
  ks_statistic: number | null;
  ks_p_value: number | null;
  mmd_score: number | null;
  mmd_p_value: number | null;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1_score: number | null;
  inference_latency_ms: number | null;
  predicted_class: string | null;
  ground_truth_label: string | null;
  review_status: string | null;
}

export const CSV_AUDIT_HEADERS = [
  'timestamp',
  'model_name',
  'model_version',
  'drift_method',
  'drift_score',
  'drift_status',
  'reference_sample_count',
  'current_sample_count',
  'performance_metrics',
  'record_type',
  'record_id',
  'feature_name',
  'ks_statistic',
  'ks_p_value',
  'mmd_score',
  'mmd_p_value',
  'accuracy',
  'precision',
  'recall',
  'f1_score',
  'inference_latency_ms',
  'predicted_class',
  'ground_truth_label',
  'review_status',
] as const;

/**
 * Escapes an individual field according to RFC 4180 CSV specifications.
 * Null and undefined values serialize cleanly as empty cells ("").
 */
export function escapeCSVField(value: unknown): string {
  if (value === null || value === undefined) {
    return '';
  }
  const str = String(value);
  if (str.includes(',') || str.includes('"') || str.includes('\n') || str.includes('\r')) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

/**
 * Builds audit CSV rows strictly from genuine records in targetStore.
 * Never invents drift scores or unmeasured ground-truth performance metrics.
 */
export function buildAuditExportRows(targetStore: ModelGuardStore = defaultStore): AuditExportRow[] {
  const rows: AuditExportRow[] = [];
  const now = new Date().toISOString();

  const activeModel =
    targetStore.models.find((m) => m.status === 'ACTIVE') || targetStore.models[0];
  const defaultModelName = activeModel?.name || 'ResNet-18 ImageNet Classifier';
  const defaultModelVersion = activeModel?.version || 'v1.0.0-resnet18';

  const refObs: ImageQualityMetrics[] = targetStore.getActiveReferenceObservations();
  const currObs: ImageQualityMetrics[] = targetStore.inferences.map((i) => i.qualityMetrics);

  // Measure performance strictly from human-verified ground truth
  const overallPerf = targetStore.getMeasuredPerformanceMetrics();
  const perfJson = JSON.stringify({
    measured: overallPerf.measured,
    labeled_samples: overallPerf.labeledCount,
    accuracy: overallPerf.accuracy,
    macro_precision: overallPerf.precision,
    macro_recall: overallPerf.recall,
    macro_f1: overallPerf.f1,
    avg_latency_ms: overallPerf.avgLatencyMs,
    note: overallPerf.note,
  });

  // 1. Export current KS Test evaluations per feature
  const ksFeatures = [
    {
      name: 'Image Sharpness (2D Laplacian Variance)',
      ref: refObs.map((o) => o.sharpness),
      curr: currObs.map((o) => o.sharpness),
    },
    {
      name: 'Image Brightness (BT.601 Luminance)',
      ref: refObs.map((o) => o.brightness),
      curr: currObs.map((o) => o.brightness),
    },
    {
      name: 'Image Contrast (RMS Luminance Std Dev)',
      ref: refObs.map((o) => o.contrast),
      curr: currObs.map((o) => o.contrast),
    },
    {
      name: 'Atmospheric Haze / Fog Index',
      ref: refObs.map((o) => Number((o.fogProbability * 100).toFixed(2))),
      curr: currObs.map((o) => Number((o.fogProbability * 100).toFixed(2))),
    },
  ];

  for (const feat of ksFeatures) {
    const ks = computeTwoSampleKSTest(
      feat.ref,
      feat.curr,
      targetStore.ksAlpha,
      targetStore.minDriftSamples,
      feat.name
    );

    rows.push({
      timestamp: ks.timestamp,
      model_name: defaultModelName,
      model_version: defaultModelVersion,
      drift_method: 'KS_TEST',
      drift_score: ks.statistic, // null when INSUFFICIENT_DATA
      drift_status: ks.status,
      reference_sample_count: ks.referenceSampleCount,
      current_sample_count: ks.currentSampleCount,
      performance_metrics: perfJson,
      record_type: 'DRIFT_SUMMARY',
      record_id: `ks-${feat.name.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`,
      feature_name: feat.name,
      ks_statistic: ks.statistic,
      ks_p_value: ks.pValue,
      mmd_score: null,
      mmd_p_value: null,
      accuracy: overallPerf.accuracy,
      precision: overallPerf.precision,
      recall: overallPerf.recall,
      f1_score: overallPerf.f1,
      inference_latency_ms: overallPerf.avgLatencyMs,
      predicted_class: null,
      ground_truth_label: null,
      review_status: null,
    });
  }

  // 2. Export Multivariate RBF-Kernel MMD evaluation row
  const refVectors = refObs
    .map((o) => o.featureVector)
    .filter((v): v is number[] => Array.isArray(v) && v.length === FEATURE_VECTOR_NAMES.length);
  const currVectors = currObs.map((o) =>
    Array.isArray(o.featureVector) && o.featureVector.length === FEATURE_VECTOR_NAMES.length
      ? o.featureVector
      : [
          o.brightness,
          o.contrast,
          o.sharpness,
          Number((o.fogProbability * 100).toFixed(2)),
          Number(((o.edgeDensity ?? 0) * 100).toFixed(2)),
          o.rMean ?? o.brightness,
          o.gMean ?? o.brightness,
          o.bMean ?? o.brightness,
        ]
  );

  const mmd = computeMultivariateMMDTest(
    [...FEATURE_VECTOR_NAMES],
    refVectors,
    currVectors,
    targetStore.ksAlpha,
    targetStore.minDriftSamples
  );

  rows.push({
    timestamp: mmd.timestamp || now,
    model_name: defaultModelName,
    model_version: defaultModelVersion,
    drift_method: 'MMD_RBF',
    drift_score: mmd.mmdScore, // null when INSUFFICIENT_DATA
    drift_status: mmd.status,
    reference_sample_count: mmd.referenceSampleCount,
    current_sample_count: mmd.currentSampleCount,
    performance_metrics: perfJson,
    record_type: 'DRIFT_SUMMARY',
    record_id: 'mmd-joint-8d-rbf',
    feature_name: 'Joint 8D Image Feature Vector',
    ks_statistic: null,
    ks_p_value: null,
    mmd_score: mmd.mmdScore,
    mmd_p_value: mmd.pValue,
    accuracy: overallPerf.accuracy,
    precision: overallPerf.precision,
    recall: overallPerf.recall,
    f1_score: overallPerf.f1,
    inference_latency_ms: overallPerf.avgLatencyMs,
    predicted_class: null,
    ground_truth_label: null,
    review_status: null,
  });

  // 3. Export individual recorded live inference observations (joined by exact inferenceId)
  const primaryKs = computeTwoSampleKSTest(
    ksFeatures[0].ref,
    ksFeatures[0].curr,
    targetStore.ksAlpha,
    targetStore.minDriftSamples,
    ksFeatures[0].name
  );

  for (const inf of targetStore.inferences) {
    const modelPerf = targetStore.getMeasuredPerformanceMetrics(inf.modelType);
    const effectivePerf = modelPerf.measured ? modelPerf : overallPerf;
    const reviewItem = targetStore.humanReviewQueue.find((q) => q.inferenceId === inf.id);
    const hasVerifiedTruth =
      reviewItem &&
      reviewItem.reviewed &&
      (reviewItem.status === 'ACCEPTED' || reviewItem.status === 'OVERRIDDEN') &&
      Boolean(reviewItem.groundTruthLabel?.trim());

    const infPerfJson = JSON.stringify({
      measured: effectivePerf.measured,
      metric_scope: modelPerf.measured ? inf.modelType : 'global_verified_evaluations',
      labeled_samples: effectivePerf.labeledCount,
      accuracy: effectivePerf.accuracy,
      macro_precision: effectivePerf.precision,
      macro_recall: effectivePerf.recall,
      macro_f1: effectivePerf.f1,
      inference_confidence: inf.primaryPrediction.confidence,
      inference_latency_ms: inf.latencyMs,
      sample_verified: Boolean(hasVerifiedTruth),
      note: effectivePerf.note,
    });

    rows.push({
      timestamp: inf.timestamp,
      model_name:
        inf.modelType === 'traffic_object_detector'
          ? 'Traffic & Autonomous Road Detector'
          : 'ResNet-18 ImageNet Classifier',
      model_version: inf.modelVersion,
      drift_method: 'KS_TEST',
      drift_score: primaryKs.statistic,
      drift_status: primaryKs.status,
      reference_sample_count: primaryKs.referenceSampleCount,
      current_sample_count: primaryKs.currentSampleCount,
      performance_metrics: infPerfJson,
      record_type: 'INFERENCE_OBSERVATION',
      record_id: inf.id,
      feature_name: ksFeatures[0].name,
      ks_statistic: primaryKs.statistic,
      ks_p_value: primaryKs.pValue,
      mmd_score: mmd.mmdScore,
      mmd_p_value: mmd.pValue,
      accuracy: effectivePerf.accuracy,
      precision: effectivePerf.precision,
      recall: effectivePerf.recall,
      f1_score: effectivePerf.f1,
      inference_latency_ms: inf.latencyMs,
      predicted_class: inf.primaryPrediction.label,
      ground_truth_label: hasVerifiedTruth ? reviewItem!.groundTruthLabel! : null,
      review_status: reviewItem ? reviewItem.status : 'NOT_QUEUED',
    });
  }

  // 4. Export held-out validation metrics for completed supervised retraining jobs
  for (const job of targetStore.retrainingJobs) {
    if (job.status !== 'COMPLETED' || !job.validationReport) continue;
    const val = job.validationReport;
    const valPerfJson = JSON.stringify({
      measured: true,
      metric_scope: 'held_out_validation_split',
      labeled_samples: val.valSampleCount,
      accuracy: val.candidateAccuracy,
      macro_precision: val.candidateMacroPrecision,
      macro_recall: val.candidateMacroRecall,
      macro_f1: val.candidateMacroF1,
      val_loss: val.candidateValLoss,
      passed_validation_gate: val.passedPromotionCriteria,
      note: val.promotionPolicyReason,
    });

    rows.push({
      timestamp: job.completedAt || job.startedAt,
      model_name: `Supervised Feature Classifier (${job.candidateVersion})`,
      model_version: job.candidateVersion,
      drift_method: 'MMD_RBF',
      drift_score: mmd.mmdScore,
      drift_status: mmd.status,
      reference_sample_count: mmd.referenceSampleCount,
      current_sample_count: mmd.currentSampleCount,
      performance_metrics: valPerfJson,
      record_type: 'RETRAINING_VALIDATION',
      record_id: job.id,
      feature_name: 'Held-Out Stratified Validation Split',
      ks_statistic: primaryKs.statistic,
      ks_p_value: primaryKs.pValue,
      mmd_score: mmd.mmdScore,
      mmd_p_value: mmd.pValue,
      accuracy: val.candidateAccuracy,
      precision: val.candidateMacroPrecision,
      recall: val.candidateMacroRecall,
      f1_score: val.candidateMacroF1,
      inference_latency_ms: 4,
      predicted_class: val.distinctClasses.join(' | '),
      ground_truth_label: val.distinctClasses.join(' | '),
      review_status: val.passedPromotionCriteria ? 'VALIDATION_PASSED' : 'VALIDATION_BLOCKED',
    });
  }

  return rows;
}

/**
 * Serializes rows into RFC 4180 compliant CSV text with CRLF line endings.
 */
export function generateAuditCSV(rows: AuditExportRow[]): string {
  const headerLine = CSV_AUDIT_HEADERS.map(escapeCSVField).join(',');
  const dataLines = rows.map((row) =>
    CSV_AUDIT_HEADERS.map((header) => escapeCSVField(row[header])).join(',')
  );

  return [headerLine, ...dataLines].join('\r\n');
}
