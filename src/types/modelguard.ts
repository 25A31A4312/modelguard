export type ModelType = 'resnet18_classifier' | 'traffic_object_detector';

export interface BoundingBox {
  ymin: number; // 0 to 1
  xmin: number; // 0 to 1
  ymax: number; // 0 to 1
  xmax: number; // 0 to 1
  label: string;
  confidence: number;
}

export interface ImageQualityMetrics {
  brightness: number; // 0 to 255 (BT.601 luminance mean)
  contrast: number; // 0 to 100 (normalized RMS luminance std dev)
  sharpness: number; // 0 to 100 (2D Laplacian variance score)
  fogProbability: number; // 0 to 1 (dark-channel haze score)
  edgeDensity?: number; // 0 to 1 (Sobel gradient magnitude ratio)
  rMean?: number; // 0 to 255
  gMean?: number; // 0 to 255
  bMean?: number; // 0 to 255
  featureVector?: number[]; // [brightness, contrast, sharpness, fog*100, edgeDensity*100, rMean, gMean, bMean]
  resolution: { width: number; height: number };
}

export interface PredictionTopClass {
  className: string;
  confidence: number;
  classId: number;
}

export interface InferenceResult {
  id: string;
  timestamp: string;
  modelType: ModelType;
  modelVersion: string;
  primaryPrediction: {
    label: string;
    confidence: number;
    category: string;
  };
  topClasses: PredictionTopClass[];
  detectedObjects: BoundingBox[];
  qualityMetrics: ImageQualityMetrics;
  driftFlags: {
    isLowConfidence: boolean;
    isDriftSuspect: boolean;
    reason?: string;
  };
  latencyMs: number;
  imageThumbnailUrl?: string;
  mlflowRunId?: string | null;
}

export type DriftMethod = 'KS_TEST' | 'MMD_RBF' | 'PSI_WASSERSTEIN';
export type DriftStatus = 'STABLE' | 'MODERATE_DRIFT' | 'SEVERE_DRIFT' | 'INSUFFICIENT_DATA';

export interface KSTestResult {
  method: 'KS_TEST';
  featureName: string;
  statistic: number | null;
  pValue: number | null;
  criticalValue: number | null;
  alpha: number;
  referenceSampleCount: number;
  currentSampleCount: number;
  minSamplesRequired: number;
  baselineMean: number | null;
  currentMean: number | null;
  status: DriftStatus;
  reason: string;
  timestamp: string;
}

export interface MMDTestResult {
  method: 'MMD_RBF';
  featureNames: string[];
  kernel: 'RBF_GAUSSIAN';
  bandwidthSigma: number | null;
  permutations: number;
  mmdScore: number | null;
  unbiasedMmdSquared: number | null;
  pValue: number | null;
  threshold: number | null;
  alpha: number;
  referenceSampleCount: number;
  currentSampleCount: number;
  minSamplesRequired: number;
  status: DriftStatus;
  reason: string;
  timestamp: string;
}

export interface DistanceMetricResult {
  method: 'PSI_WASSERSTEIN';
  featureName: string;
  baselineMean: number | null;
  currentMean: number | null;
  wassersteinDistance: number | null;
  psi: number | null;
  psiThreshold: number;
  referenceSampleCount: number;
  currentSampleCount: number;
  minSamplesRequired: number;
  status: DriftStatus;
  reason: string;
  timestamp: string;
}

export interface DriftMetric {
  featureName: string;
  method: DriftMethod;
  baselineMean: number | null;
  currentMean: number | null;
  referenceSampleCount: number;
  currentSampleCount: number;
  minSamplesRequired: number;
  ksStatistic: number | null;
  ksPValue: number | null;
  ksCriticalValue?: number | null;
  mmdScore: number | null;
  mmdPValue?: number | null;
  mmdThreshold: number | null;
  wassersteinDistance: number | null;
  psi: number | null;
  status: DriftStatus;
  threshold: number;
  alpha: number;
  interpretation: string;
  reason: string;
  timestamp: string;
}

export interface DriftTriggerEvent {
  id: string;
  timestamp: string;
  driftSignature: string;
  breachedFeatures: string[];
  ksMinPValue: number | null;
  mmdScore: number | null;
  mmdThreshold: number | null;
  eligibleLabeledSamples: number;
  minRequiredLabeledSamples: number;
  outcome:
    | 'RETRAINING_ENQUEUED'
    | 'SKIPPED_INSUFFICIENT_LABELED_DATA'
    | 'SKIPPED_DUPLICATE_EVENT'
    | 'SKIPPED_ACTIVE_JOB';
  reason: string;
  retrainingJobId?: string;
  mlflowRunId?: string | null;
}

export interface DriftAutoTriggerStatus {
  enabled: boolean;
  ksAlphaThreshold: number;
  mmdAlphaThreshold: number;
  minDriftSamplesRequired: number;
  minRetrainLabeledSamplesRequired: number;
  eligibleLabeledSamplesCount: number;
  eligibleDistinctClassesCount: number;
  lastEvent: DriftTriggerEvent | null;
  history: DriftTriggerEvent[];
}

export interface MLflowConnectionStatus {
  configured: boolean;
  connected: boolean;
  trackingUri: string | null;
  experimentId: string | null;
  experimentName: string;
  lastCheckedAt: string;
  error: string | null;
}

export interface DriftEvaluationReport {
  overallDriftState: 'HEALTHY' | 'MODERATE_DRIFT' | 'CRITICAL_DRIFT' | 'INSUFFICIENT_DATA';
  reason: string;
  lastCalculated: string;
  inferencesEvaluated: number;
  referenceDatasetId: string;
  referenceDatasetName: string;
  referenceSampleCount: number;
  currentSampleCount: number;
  minSamplesRequired: number;
  alpha: number;
  ksTests: KSTestResult[];
  mmdTest: MMDTestResult;
  distanceMetrics: DistanceMetricResult[];
  metrics: DriftMetric[];
  autoTrigger: DriftAutoTriggerStatus;
  mlflowStatus: MLflowConnectionStatus;
}

export type AlertType = 'LOW_CONFIDENCE' | 'DATA_DRIFT' | 'WEATHER_ANOMALY' | 'SYSTEM_LATENCY';
export type AlertSeverity = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'INFO';
export type AlertStatus = 'OPEN' | 'INVESTIGATING' | 'RESOLVED' | 'DISMISSED';

export interface AlertItem {
  id: string;
  timestamp: string;
  type: AlertType;
  severity: AlertSeverity;
  status: AlertStatus;
  title: string;
  description: string;
  sampleId?: string;
  metricValue?: number;
  threshold?: number;
  suggestedAction: string;
}

export interface HumanReviewSample {
  id: string;
  inferenceId?: string;
  timestamp: string;
  imageUrl: string;
  imageStorageType?: 'PERSISTENT_FILE' | 'SAMPLE_ASSET' | 'DATA_URL';
  modelType?: ModelType;
  modelVersion?: string;
  modelPrediction: string;
  modelConfidence: number;
  detectedCount: number;
  driftReason: string;
  featureVector?: number[];
  groundTruthLabel?: string;
  reviewed: boolean;
  reviewedBy?: string;
  reviewedAt?: string;
  status: 'PENDING' | 'ACCEPTED' | 'OVERRIDDEN' | 'REJECTED';
  sendToRetraining: boolean;
  retrainingJobId?: string;
}

export interface ModelVersion {
  id: string;
  name: string;
  architecture: string;
  version: string;
  status: 'ACTIVE' | 'CANDIDATE' | 'ARCHIVED';
  accuracy: number | null;
  mAP: number | null;
  macroPrecision?: number | null;
  macroRecall?: number | null;
  macroF1?: number | null;
  avgLatencyMs: number;
  trainingSamples: number;
  validationSamples?: number;
  driftToleranceScore: number;
  deployedAt: string;
  changelog: string;
  mlflowRunId?: string | null;
  passedValidation?: boolean;
  checkpointPath?: string;
}

export interface RetrainingValidationReport {
  baselineAccuracy: number;
  candidateAccuracy: number;
  baselineMacroPrecision: number;
  candidateMacroPrecision: number;
  baselineMacroRecall: number;
  candidateMacroRecall: number;
  baselineMacroF1: number;
  candidateMacroF1: number;
  baselineValLoss: number;
  candidateValLoss: number;
  valSampleCount: number;
  distinctClasses: string[];
  passedPromotionCriteria: boolean;
  promotionPolicyReason: string;
  // Backward-compatible fields mapped from macroF1 / precision
  baselineMAP: number;
  candidateMAP: number;
}

export interface RetrainingJob {
  id: string;
  triggerSource: 'MANUAL' | 'AUTO_DRIFT_THRESHOLD' | 'HUMAN_REVIEW_BATCH';
  startedAt: string;
  completedAt?: string;
  durationMs?: number;
  status: 'IDLE' | 'QUEUED' | 'TRAINING' | 'VALIDATING' | 'COMPLETED' | 'FAILED';
  currentEpoch: number;
  totalEpochs: number;
  progressPercent: number;
  trainLoss: number | null;
  valLoss: number | null;
  mAPDelta: number | null;
  accuracyDelta: number | null;
  newSamplesCount: number;
  trainSamplesCount?: number;
  valSamplesCount?: number;
  minRequiredSamples?: number;
  candidateVersion: string;
  candidateModelId?: string;
  mlflowRunId?: string | null;
  failureReason?: string;
  errorMessage?: string;
  lossHistory?: { epoch: number; trainLoss: number; valLoss: number }[];
  validationReport?: RetrainingValidationReport;
}

export interface ReferenceDataset {
  id: string;
  name: string;
  domain: string;
  sampleCount: number;
  sourceDescription?: string;
  lastUpdated: string;
  meanBrightness: number;
  meanContrast: number;
  meanSharpness: number;
  commonClasses: string[];
}

export interface MLflowRun {
  runId: string;
  experimentId: string;
  runName: string;
  status: 'RUNNING' | 'FINISHED' | 'FAILED' | 'KILLED' | 'SCHEDULED';
  startTime: number;
  endTime?: number;
  params: Record<string, string>;
  metrics: Record<string, number>;
  tags: Record<string, string>;
  artifactUri: string;
}

export interface MLflowExperiment {
  experimentId: string;
  name: string;
  artifactLocation: string;
  lifecycleStage: string;
  creationTime: number;
  lastUpdateTime: number;
}
