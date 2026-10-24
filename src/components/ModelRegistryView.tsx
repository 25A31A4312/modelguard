import React, { useState, useEffect } from 'react';
import {
  Cpu,
  ArrowUpRight,
  ShieldCheck,
  GitBranch,
  Terminal,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react';
import type {
  MLflowConnectionStatus,
  MLflowRun,
  ModelVersion,
} from '../types/modelguard.ts';
import { api } from '../services/api.ts';

interface ModelRegistryViewProps {
  models: ModelVersion[];
  onModelPromoted: () => void;
}

export const ModelRegistryView: React.FC<ModelRegistryViewProps> = ({
  models,
  onModelPromoted,
}) => {
  const [activeTab, setActiveTab] = useState<'REGISTRY' | 'MLFLOW'>('REGISTRY');
  const [mlflowStatus, setMlflowStatus] = useState<MLflowConnectionStatus | null>(null);
  const [mlflowRuns, setMlflowRuns] = useState<MLflowRun[]>([]);
  const [isLoadingRuns, setIsLoadingRuns] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const loadMlflowData = () => {
    setIsLoadingRuns(true);
    api
      .getMLflowRuns()
      .then((res) => {
        setMlflowStatus(res.status || null);
        setMlflowRuns(res.runs || []);
      })
      .catch((err) => console.error('Failed to query MLflow server:', err))
      .finally(() => setIsLoadingRuns(false));
  };

  useEffect(() => {
    if (activeTab === 'MLFLOW') {
      loadMlflowData();
    }
  }, [activeTab]);

  const handlePromote = async (modelId: string) => {
    setActionError(null);
    try {
      await api.triggerModelAction(modelId, 'promote');
      onModelPromoted();
    } catch (err: any) {
      setActionError(err.message || 'Failed to promote model');
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                <Cpu className="w-4 h-4" />
              </span>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Model Registry &amp; MLflow Experiment Tracking
              </h2>
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-2xl">
              Validation-gated candidate model promotion, persistent checkpoint artifacts, and live REST API v2.0 tracking against your configured MLflow server (<code className="text-slate-300">MLFLOW_TRACKING_URI</code>).
            </p>
          </div>

          <div className="flex items-center p-1 bg-slate-950 border border-slate-800 rounded-xl">
            <button
              onClick={() => setActiveTab('REGISTRY')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                activeTab === 'REGISTRY'
                  ? 'bg-slate-800 text-emerald-400 border border-slate-700 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Model Registry ({models.length})
            </button>
            <button
              onClick={() => setActiveTab('MLFLOW')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 cursor-pointer ${
                activeTab === 'MLFLOW'
                  ? 'bg-slate-800 text-indigo-400 border border-slate-700 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <GitBranch className="w-3.5 h-3.5" />
              <span>MLflow Tracking</span>
            </button>
          </div>
        </div>
      </div>

      {actionError && (
        <div className="p-3.5 rounded-xl bg-rose-950/40 border border-rose-900 text-rose-300 text-xs flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <span>{actionError}</span>
        </div>
      )}

      {/* View 1: Model Registry */}
      {activeTab === 'REGISTRY' && (
        <div className="space-y-4">
          {models.map((model) => {
            const isActive = model.status === 'ACTIVE';
            const isCandidate = model.status === 'CANDIDATE';
            const blockedByGate = model.passedValidation === false;

            return (
              <div
                key={model.id}
                className={`p-5 rounded-2xl border transition-all space-y-4 ${
                  isActive
                    ? 'bg-slate-900/90 border-emerald-500/50 shadow-lg shadow-emerald-950/20'
                    : isCandidate
                    ? 'bg-slate-900/60 border-indigo-500/40'
                    : 'bg-slate-950/40 border-slate-800/60 opacity-80'
                }`}
              >
                <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <div
                      className={`w-10 h-10 rounded-xl flex items-center justify-center font-bold text-sm ${
                        isActive
                          ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                          : isCandidate
                          ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40'
                          : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      <Cpu className="w-5 h-5" />
                    </div>

                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="text-base font-bold text-white">{model.name}</h3>
                        <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-950 text-slate-300 border border-slate-800">
                          {model.version}
                        </span>
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold tracking-wide uppercase ${
                            isActive
                              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                              : isCandidate
                              ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                              : 'bg-slate-800 text-slate-400'
                          }`}
                        >
                          {model.status}
                        </span>
                        {blockedByGate && (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-300 border border-rose-500/30">
                            VALIDATION GATE FAILED
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Architecture: <span className="text-slate-300">{model.architecture}</span> •
                        Registered: {new Date(model.deployedAt).toLocaleDateString()}
                        {model.mlflowRunId && (
                          <span className="ml-2 font-mono text-[11px] text-indigo-400">
                            • MLflow Run: {model.mlflowRunId}
                          </span>
                        )}
                      </p>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2">
                    {isCandidate && (
                      <button
                        onClick={() => handlePromote(model.id)}
                        disabled={blockedByGate}
                        title={
                          blockedByGate
                            ? 'Candidate failed held-out validation criteria and cannot be promoted'
                            : 'Promote validated candidate model to ACTIVE'
                        }
                        className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-emerald-500 hover:bg-emerald-600 text-slate-950 text-xs font-semibold shadow-md shadow-emerald-500/20 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        <ArrowUpRight className="w-3.5 h-3.5" />
                        <span>
                          {blockedByGate ? 'Blocked by Validation Gate' : 'Promote to Production'}
                        </span>
                      </button>
                    )}
                    {isActive && (
                      <span className="flex items-center gap-1 text-xs text-emerald-400 font-semibold bg-emerald-950/40 px-3 py-1.5 rounded-xl border border-emerald-800/40">
                        <ShieldCheck className="w-4 h-4" />
                        Active In Production
                      </span>
                    )}
                  </div>
                </div>

                {/* Metrics Row */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 text-xs">
                  <div>
                    <span className="text-slate-500 block text-[10px]">HELD-OUT ACCURACY</span>
                    <span className="font-mono font-bold text-slate-200">
                      {model.accuracy !== null ? `${(model.accuracy * 100).toFixed(1)}%` : 'Unmeasured'}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">MACRO F1 / mAP</span>
                    <span className="font-mono font-bold text-emerald-400">
                      {model.mAP !== null ? `${(model.mAP * 100).toFixed(1)}%` : 'Unmeasured'}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">TRAIN / VAL SAMPLES</span>
                    <span className="font-mono font-bold text-slate-200">
                      {model.trainingSamples}
                      {model.validationSamples != null ? ` / ${model.validationSamples}` : ''}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">AVG LATENCY</span>
                    <span className="font-mono font-bold text-indigo-400">
                      {model.avgLatencyMs}ms
                    </span>
                  </div>
                </div>

                <div className="text-[11px] text-slate-400 bg-slate-950/40 px-3 py-2 rounded-lg border border-slate-800/60">
                  <strong className="text-slate-300">Release Notes:</strong> {model.changelog}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* View 2: Genuine MLflow Tracking Server Connection & Runs */}
      {activeTab === 'MLFLOW' && (
        <div className="space-y-4">
          <div
            className={`p-4 rounded-xl border text-xs flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
              mlflowStatus?.connected
                ? 'bg-emerald-950/20 border-emerald-800/50 text-emerald-200'
                : 'bg-amber-950/30 border-amber-900/50 text-amber-200'
            }`}
          >
            <div className="flex items-start gap-2.5">
              <Terminal className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <div>
                  MLflow Tracking URI:{' '}
                  <strong className="font-mono text-white">
                    {mlflowStatus?.trackingUri || 'Not Configured (MLFLOW_TRACKING_URI unset)'}
                  </strong>
                </div>
                {mlflowStatus?.connected ? (
                  <p className="text-[11px] text-emerald-300">
                    Connected to live MLflow Tracking Server (Experiment ID:{' '}
                    <code>{mlflowStatus.experimentId}</code> —{' '}
                    <code>{mlflowStatus.experimentName}</code>).
                  </p>
                ) : (
                  <p className="text-[11px] text-amber-200/90">
                    {mlflowStatus?.error ||
                      'MLflow Tracking Server is currently unavailable. Start MLflow via `docker compose up` or set MLFLOW_TRACKING_URI=http://localhost:5000. No fake run history is displayed.'}
                  </p>
                )}
              </div>
            </div>

            <button
              onClick={loadMlflowData}
              disabled={isLoadingRuns}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 text-xs font-semibold shrink-0 cursor-pointer"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoadingRuns ? 'animate-spin' : ''}`} />
              <span>Probe Server</span>
            </button>
          </div>

          {isLoadingRuns ? (
            <div className="p-8 text-center text-xs text-slate-400">
              Querying MLflow REST API v2.0...
            </div>
          ) : !mlflowStatus?.connected ? (
            <div className="p-8 text-center bg-slate-900 border border-slate-800 rounded-2xl text-xs text-slate-400 space-y-2">
              <p className="font-semibold text-slate-200">
                MLflow Tracking Backend Unavailable
              </p>
              <p className="max-w-lg mx-auto text-slate-500">
                ModelGuard only displays authentic runs returned by a live MLflow Tracking Server. Start the bundled MLflow service with <code className="text-slate-300">docker compose up -d mlflow</code> and set <code className="text-slate-300">MLFLOW_TRACKING_URI=http://localhost:5000</code>.
              </p>
            </div>
          ) : mlflowRuns.length === 0 ? (
            <div className="p-8 text-center bg-slate-900 border border-slate-800 rounded-2xl text-xs text-slate-500">
              Connected to MLflow server — 0 runs logged in this experiment yet. Run an inference, drift check, or retraining job to log runs.
            </div>
          ) : (
            mlflowRuns.map((run) => (
              <div
                key={run.runId}
                className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-3"
              >
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-bold text-white text-sm">{run.runName}</span>
                    <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-950 text-indigo-300 border border-indigo-900/50">
                      ID: {run.runId}
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        run.status === 'FINISHED'
                          ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                          : run.status === 'RUNNING'
                          ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                          : 'bg-rose-500/20 text-rose-300'
                      }`}
                    >
                      {run.status}
                    </span>
                  </div>

                  <span className="text-xs text-slate-500 font-mono">
                    {new Date(run.startTime).toLocaleString()}
                  </span>
                </div>

                {Object.keys(run.params || {}).length > 0 && (
                  <div>
                    <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-1.5">
                      Logged Parameters
                    </span>
                    <div className="flex flex-wrap gap-1.5">
                      {Object.entries(run.params).map(([key, val]) => (
                        <span
                          key={key}
                          className="px-2 py-0.5 rounded text-[11px] bg-slate-950 text-slate-300 border border-slate-800 font-mono"
                        >
                          <span className="text-slate-500">{key}:</span> {val}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {Object.keys(run.metrics || {}).length > 0 && (
                  <div>
                    <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-1.5">
                      Logged Metrics
                    </span>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                      {Object.entries(run.metrics).map(([key, val]) => (
                        <div
                          key={key}
                          className="p-2 rounded-lg bg-slate-950 border border-slate-800 text-xs"
                        >
                          <span className="text-slate-500 text-[10px] block truncate">{key}</span>
                          <span className="font-mono font-bold text-emerald-400">
                            {typeof val === 'number' ? val.toFixed(4) : val}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="text-[11px] text-slate-500 font-mono flex items-center justify-between pt-2 border-t border-slate-800/80">
                  <span className="truncate max-w-md">Artifacts: {run.artifactUri || 'N/A'}</span>
                  <span>Workflow: {run.tags?.['modelguard.workflow'] || 'MLFLOW_RUN'}</span>
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
};
