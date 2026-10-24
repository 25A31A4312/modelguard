import type {
  MLflowConnectionStatus,
  MLflowExperiment,
  MLflowRun,
} from '../src/types/modelguard.ts';

const DEFAULT_EXPERIMENT_NAME =
  process.env.MLFLOW_EXPERIMENT_NAME || 'modelguard-cv-drift-monitoring';
const REQUEST_TIMEOUT_MS = 3000;

export interface MLflowLogOutcome {
  logged: boolean;
  runId: string | null;
  experimentId: string | null;
  artifactLogged?: boolean;
  error: string | null;
}

/**
 * Genuine MLflow Tracking Server REST API v2.0 Client.
 * Connects strictly to a real MLflow Tracking Server specified by MLFLOW_TRACKING_URI.
 * Never fabricates run IDs, experiments, or successful statuses when the server is unconfigured or unreachable.
 */
export class MLflowTracker {
  private overrideUri: string | undefined;
  private cachedExperimentId: string | null = null;
  private lastStatus: MLflowConnectionStatus = {
    configured: Boolean(process.env.MLFLOW_TRACKING_URI?.trim()),
    connected: false,
    trackingUri: process.env.MLFLOW_TRACKING_URI?.trim() || null,
    experimentId: null,
    experimentName: DEFAULT_EXPERIMENT_NAME,
    lastCheckedAt: new Date().toISOString(),
    error: process.env.MLFLOW_TRACKING_URI?.trim()
      ? 'Not probed yet'
      : 'MLFLOW_TRACKING_URI is not configured. In Docker Compose it defaults to http://mlflow:5000; for local bare-metal runs set MLFLOW_TRACKING_URI to your MLflow server URL.',
  };

  public setTrackingUri(uri: string | undefined) {
    this.overrideUri = uri;
    this.cachedExperimentId = null;
  }

  public getTrackingUri(): string | null {
    const raw = (this.overrideUri !== undefined ? this.overrideUri : process.env.MLFLOW_TRACKING_URI) || '';
    const trimmed = raw.trim().replace(/\/+$/, '');
    return trimmed.length > 0 ? trimmed : null;
  }

  private async requestJson<T>(
    endpointPath: string,
    options: RequestInit = {}
  ): Promise<{ ok: true; data: T } | { ok: false; error: string; status?: number }> {
    const baseUri = this.getTrackingUri();
    if (!baseUri) {
      return {
        ok: false,
        error:
          'MLFLOW_TRACKING_URI is not configured. Configure MLFLOW_TRACKING_URI to point to a running MLflow Tracking Server.',
      };
    }

    const url = `${baseUri}${endpointPath}`;
    try {
      const res = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...(options.headers || {}),
        },
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });

      const text = await res.text();
      if (!res.ok) {
        return {
          ok: false,
          status: res.status,
          error: `MLflow HTTP ${res.status} on ${endpointPath}: ${text.slice(0, 240)}`,
        };
      }

      const data = text ? (JSON.parse(text) as T) : ({} as T);
      return { ok: true, data };
    } catch (err: any) {
      const msg = err?.cause?.message || err?.message || String(err);
      return {
        ok: false,
        error: `MLflow server unreachable at ${baseUri}: ${msg}`,
      };
    }
  }

  public async checkConnection(
    experimentName: string = DEFAULT_EXPERIMENT_NAME
  ): Promise<MLflowConnectionStatus> {
    const uri = this.getTrackingUri();
    const now = new Date().toISOString();

    if (!uri) {
      this.lastStatus = {
        configured: false,
        connected: false,
        trackingUri: null,
        experimentId: null,
        experimentName,
        lastCheckedAt: now,
        error:
          'MLFLOW_TRACKING_URI environment variable is not set. In Docker Compose this defaults to http://mlflow:5000; for local runs point MLFLOW_TRACKING_URI to a running MLflow server.',
      };
      return this.lastStatus;
    }

    const expRes = await this.ensureExperiment(experimentName);
    if (!expRes.ok) {
      this.lastStatus = {
        configured: true,
        connected: false,
        trackingUri: uri,
        experimentId: null,
        experimentName,
        lastCheckedAt: now,
        error: expRes.error,
      };
      return this.lastStatus;
    }

    this.lastStatus = {
      configured: true,
      connected: true,
      trackingUri: uri,
      experimentId: expRes.experimentId,
      experimentName,
      lastCheckedAt: now,
      error: null,
    };
    return this.lastStatus;
  }

  public getCachedStatus(): MLflowConnectionStatus {
    const uri = this.getTrackingUri();
    if (!uri) {
      return {
        configured: false,
        connected: false,
        trackingUri: null,
        experimentId: null,
        experimentName: DEFAULT_EXPERIMENT_NAME,
        lastCheckedAt: new Date().toISOString(),
        error:
          'MLFLOW_TRACKING_URI is not configured. Start an MLflow tracking server and set MLFLOW_TRACKING_URI.',
      };
    }
    return {
      ...this.lastStatus,
      configured: true,
      trackingUri: uri,
    };
  }

  public async ensureExperiment(
    experimentName: string = DEFAULT_EXPERIMENT_NAME
  ): Promise<{ ok: true; experimentId: string } | { ok: false; error: string }> {
    if (this.cachedExperimentId && experimentName === DEFAULT_EXPERIMENT_NAME) {
      // Quick verify experiment still exists
      const check = await this.requestJson<{ experiment?: { experiment_id: string } }>(
        `/api/2.0/mlflow/experiments/get?experiment_id=${encodeURIComponent(this.cachedExperimentId)}`
      );
      if (check.ok && check.data.experiment?.experiment_id) {
        return { ok: true, experimentId: check.data.experiment.experiment_id };
      }
      this.cachedExperimentId = null;
    }

    const getByName = await this.requestJson<{ experiment?: { experiment_id: string; lifecycle_stage?: string } }>(
      `/api/2.0/mlflow/experiments/get-by-name?experiment_name=${encodeURIComponent(experimentName)}`
    );

    if (getByName.ok && getByName.data.experiment?.experiment_id) {
      const expId = String(getByName.data.experiment.experiment_id);
      if (experimentName === DEFAULT_EXPERIMENT_NAME) this.cachedExperimentId = expId;
      return { ok: true, experimentId: expId };
    }

    // If 404 RESOURCE_DOES_NOT_EXIST, create the experiment
    if (!getByName.ok && getByName.status !== 404 && !getByName.error.includes('RESOURCE_DOES_NOT_EXIST')) {
      return { ok: false, error: getByName.error };
    }

    const createRes = await this.requestJson<{ experiment_id: string }>(
      `/api/2.0/mlflow/experiments/create`,
      {
        method: 'POST',
        body: JSON.stringify({
          name: experimentName,
          tags: [
            { key: 'project', value: 'ModelGuard' },
            { key: 'domain', value: 'computer-vision-drift-mlops' },
          ],
        }),
      }
    );

    if (!createRes.ok) {
      return { ok: false, error: createRes.error };
    }

    const expId = String(createRes.data.experiment_id);
    if (experimentName === DEFAULT_EXPERIMENT_NAME) this.cachedExperimentId = expId;
    return { ok: true, experimentId: expId };
  }

  public async createRun(
    runName: string,
    tags: Record<string, string> = {},
    experimentName: string = DEFAULT_EXPERIMENT_NAME
  ): Promise<{ ok: true; runId: string; experimentId: string; artifactUri: string } | { ok: false; error: string }> {
    const exp = await this.ensureExperiment(experimentName);
    if (!exp.ok) {
      this.lastStatus = {
        ...this.lastStatus,
        configured: Boolean(this.getTrackingUri()),
        connected: false,
        trackingUri: this.getTrackingUri(),
        lastCheckedAt: new Date().toISOString(),
        error: exp.error,
      };
      return { ok: false, error: exp.error };
    }

    const startTime = Date.now();
    const tagList = [
      { key: 'mlflow.runName', value: runName },
      ...Object.entries(tags).map(([key, value]) => ({ key, value: String(value) })),
    ];

    const res = await this.requestJson<{
      run: {
        info: {
          run_id: string;
          experiment_id: string;
          artifact_uri: string;
        };
      };
    }>(`/api/2.0/mlflow/runs/create`, {
      method: 'POST',
      body: JSON.stringify({
        experiment_id: exp.experimentId,
        run_name: runName,
        start_time: startTime,
        tags: tagList,
      }),
    });

    if (!res.ok) {
      this.lastStatus = {
        ...this.lastStatus,
        configured: true,
        connected: false,
        trackingUri: this.getTrackingUri(),
        lastCheckedAt: new Date().toISOString(),
        error: res.error,
      };
      return { ok: false, error: res.error };
    }

    this.lastStatus = {
      configured: true,
      connected: true,
      trackingUri: this.getTrackingUri(),
      experimentId: exp.experimentId,
      experimentName,
      lastCheckedAt: new Date().toISOString(),
      error: null,
    };

    return {
      ok: true,
      runId: res.data.run.info.run_id,
      experimentId: res.data.run.info.experiment_id,
      artifactUri: res.data.run.info.artifact_uri || '',
    };
  }

  public async logBatch(
    runId: string,
    params: Record<string, string | number | boolean> = {},
    metrics: Record<string, number | null | undefined> = {},
    tags: Record<string, string> = {},
    step: number = 0
  ): Promise<{ ok: true } | { ok: false; error: string }> {
    const timestamp = Date.now();

    const mlflowParams = Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== null)
      .map(([key, value]) => ({
        key: key.slice(0, 250),
        value: String(value).slice(0, 500),
      }));

    const mlflowMetrics = Object.entries(metrics)
      .filter((entry): entry is [string, number] => typeof entry[1] === 'number' && Number.isFinite(entry[1]))
      .map(([key, value]) => ({
        key: key.slice(0, 250),
        value,
        timestamp,
        step,
      }));

    const mlflowTags = Object.entries(tags).map(([key, value]) => ({
      key: key.slice(0, 250),
      value: String(value).slice(0, 5000),
    }));

    if (mlflowParams.length === 0 && mlflowMetrics.length === 0 && mlflowTags.length === 0) {
      return { ok: true };
    }

    const res = await this.requestJson(`/api/2.0/mlflow/runs/log-batch`, {
      method: 'POST',
      body: JSON.stringify({
        run_id: runId,
        params: mlflowParams,
        metrics: mlflowMetrics,
        tags: mlflowTags,
      }),
    });

    if (!res.ok) return { ok: false, error: res.error };
    return { ok: true };
  }

  /**
   * Uploads a JSON artifact via MLflow 2.x proxied artifact storage (`--serve-artifacts`).
   */
  public async logJsonArtifact(
    experimentId: string,
    runId: string,
    fileName: string,
    payload: unknown,
    artifactUri?: string
  ): Promise<boolean> {
    const baseUri = this.getTrackingUri();
    if (!baseUri) return false;

    const cleanFile = fileName.replace(/^\/+/, '');
    const candidateUrls: string[] = [];

    if (artifactUri && artifactUri.startsWith('mlflow-artifacts:')) {
      const subPath = artifactUri.replace(/^mlflow-artifacts:\/*/, '').replace(/\/+$/, '');
      const fullSubPath = subPath ? `${subPath}/${cleanFile}` : cleanFile;
      candidateUrls.push(`${baseUri}/api/2.0/mlflow-artifacts/artifacts/${fullSubPath}`);
    }

    candidateUrls.push(
      `${baseUri}/api/2.0/mlflow-artifacts/artifacts/${experimentId}/${runId}/artifacts/${cleanFile}`,
      `${baseUri}/api/2.0/mlflow-artifacts/artifacts/${runId}/artifacts/${cleanFile}`
    );

    const uniqueUrls = Array.from(new Set(candidateUrls));
    const body = Buffer.from(JSON.stringify(payload, null, 2), 'utf-8');
    for (const url of uniqueUrls) {
      try {
        const res = await fetch(url, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/octet-stream' },
          body,
          signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
        });
        if (res.ok) return true;
      } catch {
        // Ignore if artifact proxy endpoint is not enabled on server
      }
    }
    return false;
  }

  public async terminateRun(
    runId: string,
    status: 'FINISHED' | 'FAILED' | 'KILLED' = 'FINISHED'
  ): Promise<{ ok: true } | { ok: false; error: string }> {
    const res = await this.requestJson(`/api/2.0/mlflow/runs/update`, {
      method: 'POST',
      body: JSON.stringify({
        run_id: runId,
        status,
        end_time: Date.now(),
      }),
    });
    if (!res.ok) return { ok: false, error: res.error };
    return { ok: true };
  }

  /**
   * Logs a complete single-shot MLflow run (used for inference, drift evaluations, and drift triggers).
   */
  public async logWorkflowRun(options: {
    workflowType: 'INFERENCE' | 'DRIFT_EVALUATION' | 'DRIFT_TRIGGER' | 'RETRAINING';
    runName: string;
    params: Record<string, string | number | boolean>;
    metrics: Record<string, number | null | undefined>;
    tags?: Record<string, string>;
    artifactName?: string;
    artifactPayload?: unknown;
    finalStatus?: 'FINISHED' | 'FAILED';
  }): Promise<MLflowLogOutcome> {
    const created = await this.createRun(options.runName, {
      'modelguard.workflow': options.workflowType,
      ...(options.tags || {}),
    });

    if (!created.ok) {
      return {
        logged: false,
        runId: null,
        experimentId: null,
        error: created.error,
      };
    }

    const batchRes = await this.logBatch(created.runId, options.params, options.metrics);
    let artifactLogged = false;
    if (options.artifactName && options.artifactPayload !== undefined) {
      artifactLogged = await this.logJsonArtifact(
        created.experimentId,
        created.runId,
        options.artifactName,
        options.artifactPayload,
        created.artifactUri
      );
    }

    const termRes = await this.terminateRun(created.runId, options.finalStatus || 'FINISHED');
    if (!batchRes.ok || !termRes.ok) {
      return {
        logged: false,
        runId: created.runId,
        experimentId: created.experimentId,
        artifactLogged,
        error: (!batchRes.ok ? batchRes.error : '') || (!termRes.ok ? termRes.error : null),
      };
    }

    return {
      logged: true,
      runId: created.runId,
      experimentId: created.experimentId,
      artifactLogged,
      error: null,
    };
  }

  /**
   * Lists real experiments from the MLflow server. Never returns fabricated experiments.
   */
  public async listExperiments(): Promise<{
    status: MLflowConnectionStatus;
    experiments: MLflowExperiment[];
  }> {
    const conn = await this.checkConnection();
    if (!conn.connected) {
      return { status: conn, experiments: [] };
    }

    const res = await this.requestJson<{
      experiments?: Array<{
        experiment_id: string;
        name: string;
        artifact_location: string;
        lifecycle_stage: string;
        creation_time?: number;
        last_update_time?: number;
      }>;
    }>(`/api/2.0/mlflow/experiments/search?max_results=50`);

    if (!res.ok) {
      return {
        status: { ...conn, connected: false, error: res.error },
        experiments: [],
      };
    }

    const experiments: MLflowExperiment[] = (res.data.experiments || []).map((e) => ({
      experimentId: String(e.experiment_id),
      name: e.name,
      artifactLocation: e.artifact_location || '',
      lifecycleStage: e.lifecycle_stage || 'active',
      creationTime: Number(e.creation_time || Date.now()),
      lastUpdateTime: Number(e.last_update_time || Date.now()),
    }));

    return { status: conn, experiments };
  }

  /**
   * Searches real runs from the MLflow server. Never returns fabricated runs.
   */
  public async searchRuns(experimentId?: string): Promise<{
    status: MLflowConnectionStatus;
    runs: MLflowRun[];
  }> {
    const conn = await this.checkConnection();
    if (!conn.connected) {
      return { status: conn, runs: [] };
    }

    const targetExpIds = experimentId
      ? [experimentId]
      : conn.experimentId
      ? [conn.experimentId]
      : [];

    if (targetExpIds.length === 0) {
      return { status: conn, runs: [] };
    }

    const res = await this.requestJson<{
      runs?: Array<{
        info: {
          run_id: string;
          run_name?: string;
          experiment_id: string;
          status: MLflowRun['status'];
          start_time: number;
          end_time?: number;
          artifact_uri?: string;
        };
        data?: {
          metrics?: Array<{ key: string; value: number }>;
          params?: Array<{ key: string; value: string }>;
          tags?: Array<{ key: string; value: string }>;
        };
      }>;
    }>(`/api/2.0/mlflow/runs/search`, {
      method: 'POST',
      body: JSON.stringify({
        experiment_ids: targetExpIds,
        max_results: 50,
        order_by: ['attributes.start_time DESC'],
      }),
    });

    if (!res.ok) {
      return {
        status: { ...conn, connected: false, error: res.error },
        runs: [],
      };
    }

    const runs: MLflowRun[] = (res.data.runs || []).map((r) => {
      const params: Record<string, string> = {};
      for (const p of r.data?.params || []) params[p.key] = p.value;

      const metrics: Record<string, number> = {};
      for (const m of r.data?.metrics || []) metrics[m.key] = Number(m.value);

      const tags: Record<string, string> = {};
      for (const t of r.data?.tags || []) tags[t.key] = t.value;

      return {
        runId: r.info.run_id,
        experimentId: String(r.info.experiment_id),
        runName: r.info.run_name || tags['mlflow.runName'] || r.info.run_id,
        status: r.info.status,
        startTime: Number(r.info.start_time),
        endTime: r.info.end_time ? Number(r.info.end_time) : undefined,
        params,
        metrics,
        tags,
        artifactUri: r.info.artifact_uri || '',
      };
    });

    return { status: conn, runs };
  }
}

export const mlflowTracker = new MLflowTracker();
