import React from 'react';
import {
  Activity,
  Layers,
  AlertTriangle,
  ShieldCheck,
  Zap,
  ArrowRight,
  Clock,
  Radio,
  FileCheck,
  CheckCircle2,
  Download,
  Info,
} from 'lucide-react';
import type {
  AlertItem,
  DriftEvaluationReport,
  DriftMetric,
  InferenceResult,
  ModelVersion,
} from '../types/modelguard.ts';
import type { ActiveTab } from './Navbar.tsx';
import { api } from '../services/api.ts';

interface DashboardViewProps {
  inferences: InferenceResult[];
  driftMetrics: DriftMetric[];
  driftReport?: DriftEvaluationReport | null;
  alerts: AlertItem[];
  models: ModelVersion[];
  measuredPerformance?: {
    measured: boolean;
    labeledCount: number;
    accuracy: number | null;
    f1: number | null;
    avgLatencyMs: number | null;
  } | null;
  pendingReviewCount: number;
  setActiveTab: (tab: ActiveTab) => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  inferences,
  driftMetrics,
  driftReport,
  alerts,
  models,
  measuredPerformance,
  pendingReviewCount,
  setActiveTab,
}) => {
  const activeAlerts = alerts.filter((a) => a.status === 'OPEN');
  const criticalAlerts = activeAlerts.filter((a) => a.severity === 'CRITICAL');

  const currentCount = driftReport?.currentSampleCount ?? inferences.length;
  const minRequired = driftReport?.minSamplesRequired ?? 5;
  const isInsufficient =
    !driftReport ||
    driftReport.overallDriftState === 'INSUFFICIENT_DATA' ||
    currentCount < minRequired;

  const validKsMetrics = driftMetrics.filter((m) => m.ksStatistic !== null);
  const topDriftMetric =
    validKsMetrics.length > 0
      ? validKsMetrics.reduce((max, m) =>
          (m.ksStatistic ?? 0) > (max.ksStatistic ?? 0) ? m : max
        )
      : driftMetrics[0];

  const activeModelCount = models.filter((m) => m.status === 'ACTIVE').length;

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl relative overflow-hidden">
        <div className="absolute right-0 top-0 bottom-0 w-96 bg-emerald-500/5 blur-3xl pointer-events-none" />
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 relative z-10">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                <CheckCircle2 className="w-3.5 h-3.5" />
                Container Port 3000 Verified
              </span>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
                Two-Sample KS &amp; RBF-MMD Active
              </span>
            </div>
            <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">
              ModelGuard Production Telemetry &amp; Drift Center
            </h1>
            <p className="text-slate-400 text-xs md:text-sm mt-1 max-w-2xl">
              Real-time computer vision inference gateway, two-sample Kolmogorov–Smirnov &amp; Multivariate RBF-MMD drift detection, and human-in-the-loop retraining pipeline.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => api.downloadAuditCSV()}
              className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs md:text-sm font-semibold border border-slate-700 transition-all cursor-pointer shadow-sm"
            >
              <Download className="w-4 h-4 text-emerald-400" />
              <span>Export Audit CSV</span>
            </button>
            <button
              onClick={() => setActiveTab('inference')}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-600 text-slate-950 text-xs md:text-sm font-semibold transition-all shadow-lg shadow-emerald-500/20 cursor-pointer"
            >
              <Zap className="w-4 h-4 fill-current" />
              <span>Launch Inference Sandbox</span>
            </button>
          </div>
        </div>
      </div>

      {/* KPI Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Metric 1: Drift Monitoring Status (Honest INSUFFICIENT_DATA when < minRequired) */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 relative overflow-hidden">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium mb-2">
            <span>DRIFT EVALUATION (KS / MMD)</span>
            <Layers className="w-4 h-4 text-amber-400" />
          </div>
          {isInsufficient ? (
            <>
              <div className="flex items-baseline gap-2">
                <span className="text-lg font-bold font-mono text-amber-300">
                  INSUFFICIENT DATA
                </span>
              </div>
              <div className="mt-2 text-xs flex items-center justify-between text-slate-400">
                <span>
                  Current: <strong className="text-slate-200">{currentCount}</strong> / {minRequired} min
                </span>
                <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-300 border border-slate-700">
                  NO TEST RUN
                </span>
              </div>
            </>
          ) : (
            <>
              <div className="flex items-baseline gap-2">
                <span
                  className={`text-2xl font-bold font-mono ${
                    topDriftMetric?.status === 'SEVERE_DRIFT'
                      ? 'text-rose-400'
                      : topDriftMetric?.status === 'MODERATE_DRIFT'
                      ? 'text-amber-400'
                      : 'text-emerald-400'
                  }`}
                >
                  D={topDriftMetric?.ksStatistic?.toFixed(3) ?? '—'}
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  p={topDriftMetric?.ksPValue?.toFixed(3) ?? '—'}
                </span>
              </div>
              <div className="mt-2 text-xs flex items-center justify-between text-slate-400">
                <span className="truncate max-w-[165px]">{topDriftMetric?.featureName}</span>
                <span
                  className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                    topDriftMetric?.status === 'SEVERE_DRIFT'
                      ? 'bg-rose-500/20 text-rose-300'
                      : topDriftMetric?.status === 'MODERATE_DRIFT'
                      ? 'bg-amber-500/20 text-amber-300'
                      : 'bg-emerald-500/20 text-emerald-300'
                  }`}
                >
                  {topDriftMetric?.status.replace('_', ' ')}
                </span>
              </div>
            </>
          )}
        </div>

        {/* Metric 2: Active Alerts */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium mb-2">
            <span>ACTIVE ALERTS</span>
            <AlertTriangle className="w-4 h-4 text-rose-400" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-white">
              {activeAlerts.length}
            </span>
            <span className="text-xs text-rose-400 font-medium">
              ({criticalAlerts.length} Critical)
            </span>
          </div>
          <div className="mt-2 text-xs flex items-center justify-between text-slate-400">
            <span>Data Drift: {activeAlerts.filter((a) => a.type === 'DATA_DRIFT').length}</span>
            <span>Low Conf: {activeAlerts.filter((a) => a.type === 'LOW_CONFIDENCE').length}</span>
          </div>
        </div>

        {/* Metric 3: Human Review Pending */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium mb-2">
            <span>HUMAN REVIEW QUEUE</span>
            <FileCheck className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-white">
              {pendingReviewCount}
            </span>
            <span className="text-xs text-amber-400 font-medium">pending triage</span>
          </div>
          <div className="mt-2 text-xs flex items-center justify-between text-slate-400">
            <span>Retraining Candidates</span>
            <button
              onClick={() => setActiveTab('review')}
              className="text-indigo-400 hover:text-indigo-300 font-medium cursor-pointer"
            >
              Review Now &rarr;
            </button>
          </div>
        </div>

        {/* Metric 4: Production Model Performance (Honest Measured Status) */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium mb-2">
            <span>MODELS IN PRODUCTION</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-emerald-400">
              {activeModelCount} Active
            </span>
            <span className="text-xs text-slate-400">
              / {models.length} Registered
            </span>
          </div>
          <div className="mt-2 text-xs flex items-center justify-between text-slate-400">
            {measuredPerformance?.measured && measuredPerformance.accuracy !== null ? (
              <>
                <span>Verified Acc: {(measuredPerformance.accuracy * 100).toFixed(1)}%</span>
                <span>N={measuredPerformance.labeledCount} labeled</span>
              </>
            ) : (
              <>
                <span>Acc: Unmeasured (0 labeled)</span>
                <span>
                  {measuredPerformance?.avgLatencyMs != null
                    ? `${measuredPerformance.avgLatencyMs}ms`
                    : 'No runs'}
                </span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Main Grid: Drift Summary & Live Prediction Stream */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Drift Status Summary */}
        <div className="lg:col-span-1 bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-emerald-400" />
              <h2 className="font-semibold text-white text-sm">Feature Drift Gauges</h2>
            </div>
            <button
              onClick={() => setActiveTab('drift')}
              className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 cursor-pointer"
            >
              Deep Analysis <ArrowRight className="w-3 h-3" />
            </button>
          </div>

          {isInsufficient && (
            <div className="p-3 rounded-xl bg-amber-950/30 border border-amber-900/50 text-xs text-amber-200 flex items-start gap-2">
              <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <span>
                <strong>Insufficient Data ({currentCount}/{minRequired} samples):</strong> Statistical KS &amp; MMD scores are withheld until at least {minRequired} real inferences are recorded.
              </span>
            </div>
          )}

          <div className="space-y-3">
            {driftMetrics.map((metric) => {
              const mInsuff = metric.status === 'INSUFFICIENT_DATA' || metric.ksStatistic === null;
              return (
                <div
                  key={metric.featureName}
                  className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-1.5"
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-slate-300 truncate max-w-[175px]">
                      {metric.featureName}
                    </span>
                    <span
                      className={`font-mono font-bold text-xs ${
                        mInsuff
                          ? 'text-slate-400'
                          : metric.status === 'SEVERE_DRIFT'
                          ? 'text-rose-400'
                          : metric.status === 'MODERATE_DRIFT'
                          ? 'text-amber-400'
                          : 'text-emerald-400'
                      }`}
                    >
                      {mInsuff ? 'INSUFFICIENT DATA' : `KS D: ${metric.ksStatistic?.toFixed(3)}`}
                    </span>
                  </div>

                  <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        mInsuff
                          ? 'bg-slate-700'
                          : metric.status === 'SEVERE_DRIFT'
                          ? 'bg-rose-500'
                          : metric.status === 'MODERATE_DRIFT'
                          ? 'bg-amber-500'
                          : 'bg-emerald-500'
                      }`}
                      style={{
                        width: mInsuff
                          ? '0%'
                          : `${Math.min(100, (metric.ksStatistic ?? 0) * 100)}%`,
                      }}
                    />
                  </div>

                  <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
                    <span>
                      KS p:{' '}
                      {metric.ksPValue === null
                        ? '—'
                        : metric.ksPValue < 0.001
                        ? '<0.001'
                        : metric.ksPValue.toFixed(3)}
                    </span>
                    <span>
                      Ref: {metric.referenceSampleCount} | Curr: {metric.currentSampleCount}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Live Inferences Stream */}
        <div className="lg:col-span-2 bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Radio className="w-4 h-4 text-emerald-400 animate-pulse" />
              <h2 className="font-semibold text-white text-sm">Recent Inferences Feed</h2>
            </div>
            <span className="text-xs text-slate-400">
              {inferences.length} live observation{inferences.length === 1 ? '' : 's'} recorded
            </span>
          </div>

          {inferences.length === 0 ? (
            <div className="border border-dashed border-slate-800 rounded-xl p-8 text-center space-y-3">
              <div className="w-12 h-12 rounded-full bg-slate-800 flex items-center justify-center mx-auto text-slate-400">
                <Activity className="w-6 h-6" />
              </div>
              <h3 className="text-slate-300 font-medium text-sm">
                0 current inference observations recorded
              </h3>
              <p className="text-xs text-slate-500 max-w-sm mx-auto">
                Upload or test real images in the Inference Sandbox to record decoded image-quality observations and run KS/MMD drift evaluations.
              </p>
              <button
                onClick={() => setActiveTab('inference')}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold cursor-pointer"
              >
                Open Sandbox
              </button>
            </div>
          ) : (
            <div className="space-y-3 max-h-[460px] overflow-y-auto pr-1">
              {inferences.map((inf) => (
                <div
                  key={inf.id}
                  className={`p-3.5 rounded-xl border transition-all ${
                    inf.driftFlags.isDriftSuspect
                      ? 'bg-rose-950/20 border-rose-900/40'
                      : inf.driftFlags.isLowConfidence
                      ? 'bg-amber-950/20 border-amber-900/40'
                      : 'bg-slate-950/60 border-slate-800/80'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-3">
                      {inf.imageThumbnailUrl ? (
                        <img
                          src={inf.imageThumbnailUrl}
                          alt="thumb"
                          className="w-12 h-12 rounded-lg object-cover border border-slate-700"
                        />
                      ) : (
                        <div className="w-12 h-12 rounded-lg bg-slate-800 flex items-center justify-center text-slate-400">
                          <Activity className="w-5 h-5" />
                        </div>
                      )}
                      <div>
                        <div className="flex items-center gap-2">
                          <h4 className="text-sm font-semibold text-white capitalize">
                            {inf.primaryPrediction.label}
                          </h4>
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                            {inf.modelType === 'traffic_object_detector'
                              ? 'Object Detector'
                              : 'ResNet-18'}
                          </span>
                        </div>
                        <div className="flex items-center gap-3 text-xs text-slate-400 mt-1">
                          <span className="font-mono text-emerald-400 font-bold">
                            {(inf.primaryPrediction.confidence * 100).toFixed(1)}% conf
                          </span>
                          <span>•</span>
                          <span>Sharpness: {inf.qualityMetrics.sharpness}/100</span>
                          <span>•</span>
                          <span>
                            Fog: {(inf.qualityMetrics.fogProbability * 100).toFixed(0)}%
                          </span>
                          <span>•</span>
                          <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3 text-slate-500" />
                            {inf.latencyMs}ms
                          </span>
                        </div>
                      </div>
                    </div>

                    <div>
                      {inf.driftFlags.isDriftSuspect ? (
                        <span className="px-2 py-1 rounded-full text-[10px] font-bold bg-rose-500/20 text-rose-300 border border-rose-500/30 flex items-center gap-1">
                          <AlertTriangle className="w-3 h-3" />
                          Drift Suspect
                        </span>
                      ) : inf.driftFlags.isLowConfidence ? (
                        <span className="px-2 py-1 rounded-full text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                          Low Conf
                        </span>
                      ) : (
                        <span className="px-2 py-1 rounded-full text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
                          <CheckCircle2 className="w-3 h-3" />
                          Normal
                        </span>
                      )}
                    </div>
                  </div>

                  {inf.driftFlags.reason && (
                    <div className="mt-2 text-[11px] text-amber-300/90 bg-amber-950/30 px-2.5 py-1 rounded border border-amber-900/30">
                      Reason: {inf.driftFlags.reason}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
