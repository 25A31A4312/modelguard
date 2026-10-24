import type {
  AlertItem,
  DriftAutoTriggerStatus,
  DriftEvaluationReport,
  HumanReviewSample,
  InferenceResult,
  MLflowConnectionStatus,
  MLflowExperiment,
  MLflowRun,
  ModelType,
  ModelVersion,
  ReferenceDataset,
  RetrainingJob,
} from '../types/modelguard.ts';

export const api = {
  async checkHealth(): Promise<any> {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error('Health check failed');
    return res.json();
  },

  async getInferences(): Promise<{ inferences: InferenceResult[]; count: number }> {
    const res = await fetch('/api/inferences');
    if (!res.ok) throw new Error('Failed to fetch inferences');
    return res.json();
  },

  async runInference(
    imageBase64: string,
    mimeType: string,
    modelType: ModelType
  ): Promise<InferenceResult> {
    const res = await fetch('/api/inference', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageBase64, mimeType, modelType }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || 'Inference failed');
    }
    return res.json();
  },

  async getDriftMetrics(
    alpha?: number,
    minSamples?: number
  ): Promise<DriftEvaluationReport> {
    const params = new URLSearchParams();
    if (typeof alpha === 'number') params.set('alpha', String(alpha));
    if (typeof minSamples === 'number') params.set('minSamples', String(minSamples));
    const qs = params.toString();
    const res = await fetch(`/api/monitoring/drift${qs ? `?${qs}` : ''}`);
    if (!res.ok) throw new Error('Failed to fetch drift metrics');
    return res.json();
  },

  async updateMonitoringConfig(config: {
    minDriftSamples?: number;
    ksAlpha?: number;
    minRetrainLabeledSamples?: number;
    autoRetrainEnabled?: boolean;
    activeReferenceDatasetId?: string;
  }): Promise<DriftEvaluationReport> {
    const res = await fetch('/api/monitoring/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!res.ok) throw new Error('Failed to update monitoring configuration');
    return res.json();
  },

  async getAlerts(
    typeFilter: string = 'ALL',
    statusFilter: string = 'ALL'
  ): Promise<{
    alerts: AlertItem[];
    summary: {
      total: number;
      open: number;
      critical: number;
      dataDriftCount: number;
      lowConfidenceCount: number;
    };
  }> {
    const params = new URLSearchParams();
    if (typeFilter !== 'ALL') params.set('type', typeFilter);
    if (statusFilter !== 'ALL') params.set('status', statusFilter);

    const res = await fetch(`/api/alerts?${params.toString()}`);
    if (!res.ok) throw new Error('Failed to fetch alerts');
    return res.json();
  },

  async updateAlertStatus(id: string, status: string): Promise<AlertItem> {
    const res = await fetch(`/api/alerts/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status }),
    });
    if (!res.ok) throw new Error('Failed to update alert');
    return res.json();
  },

  async getHumanReviewQueue(): Promise<{
    queue: HumanReviewSample[];
    pendingCount: number;
    approvedForRetraining: number;
  }> {
    const res = await fetch('/api/human-review');
    if (!res.ok) throw new Error('Failed to fetch review queue');
    return res.json();
  },

  async enqueueHumanReview(
    inferenceId: string,
    reason?: string
  ): Promise<HumanReviewSample> {
    const res = await fetch('/api/human-review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ inferenceId, reason }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || 'Failed to enqueue sample for human review');
    }
    return res.json();
  },

  async updateHumanReview(
    id: string,
    updates: Partial<HumanReviewSample>
  ): Promise<HumanReviewSample> {
    const res = await fetch(`/api/human-review/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    if (!res.ok) throw new Error('Failed to update review item');
    return res.json();
  },

  async getModels(): Promise<{
    models: ModelVersion[];
    measuredPerformance?: {
      measured: boolean;
      labeledCount: number;
      accuracy: number | null;
      precision: number | null;
      recall: number | null;
      f1: number | null;
      avgLatencyMs: number | null;
      note: string;
    };
  }> {
    const res = await fetch('/api/models');
    if (!res.ok) throw new Error('Failed to fetch models');
    return res.json();
  },

  async triggerModelAction(id: string, action: 'promote' | 'rollback'): Promise<any> {
    const res = await fetch(`/api/models/${id}/action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || 'Failed to perform model action');
    }
    return data;
  },

  async getRetrainingJobs(): Promise<{
    jobs: RetrainingJob[];
    activeJob: RetrainingJob | null;
    eligibility?: {
      eligibleLabeledSamplesCount: number;
      eligibleDistinctClassesCount: number;
      minRequiredSamples: number;
      readyToTrain: boolean;
    };
    autoTrigger?: DriftAutoTriggerStatus;
  }> {
    const res = await fetch('/api/retrain');
    if (!res.ok) throw new Error('Failed to fetch retraining jobs');
    return res.json();
  },

  async triggerRetrain(triggerSource: string = 'MANUAL'): Promise<RetrainingJob> {
    const res = await fetch('/api/retrain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ triggerSource }),
    });
    const data = await res.json().catch(() => ({}));
    // HTTP 422 returns a structured FAILED RetrainingJob record (e.g. insufficient labeled data)
    if (res.status === 422 && data && data.id) {
      return data as RetrainingJob;
    }
    if (!res.ok) {
      throw new Error(data.error || 'Failed to trigger retraining');
    }
    return data as RetrainingJob;
  },

  async getReferenceDatasets(): Promise<{
    datasets: ReferenceDataset[];
    activeReferenceDatasetId?: string;
  }> {
    const res = await fetch('/api/reference-datasets');
    if (!res.ok) throw new Error('Failed to fetch reference datasets');
    return res.json();
  },

  async getMLflowStatus(): Promise<MLflowConnectionStatus> {
    const res = await fetch('/api/mlflow/status');
    if (!res.ok) throw new Error('Failed to probe MLflow status');
    return res.json();
  },

  async getMLflowRuns(experimentId?: string): Promise<{
    status: MLflowConnectionStatus;
    runs: MLflowRun[];
  }> {
    const url = experimentId
      ? `/api/mlflow/runs?experimentId=${encodeURIComponent(experimentId)}`
      : '/api/mlflow/runs';
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to fetch MLflow runs');
    return res.json();
  },

  async getMLflowExperiments(): Promise<{
    status: MLflowConnectionStatus;
    experiments: MLflowExperiment[];
  }> {
    const res = await fetch('/api/mlflow/experiments');
    if (!res.ok) throw new Error('Failed to fetch MLflow experiments');
    return res.json();
  },

  async downloadAuditCSV(): Promise<void> {
    const res = await fetch('/api/export/csv');
    if (!res.ok) throw new Error('Failed to export CSV');
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `modelguard_audit_export_${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  },
};
