import React, { useState } from 'react';
import {
  Layers,
  HelpCircle,
  Download,
  Info,
  GitBranch,
  RefreshCw,
} from 'lucide-react';
import type { DriftEvaluationReport, DriftMetric } from '../types/modelguard.ts';
import { api } from '../services/api.ts';

interface DriftMonitoringViewProps {
  driftMetrics: DriftMetric[];
  driftReport?: DriftEvaluationReport | null;
  inferencesCount: number;
  onReportUpdated?: (report: DriftEvaluationReport) => void;
}

export const DriftMonitoringView: React.FC<DriftMonitoringViewProps> = ({
  driftMetrics,
  driftReport,
  inferencesCount,
  onReportUpdated,
}) => {
  const [selectedAlpha, setSelectedAlpha] = useState<number>(driftReport?.alpha ?? 0.05);
  const [minSamplesInput, setMinSamplesInput] = useState<number>(
    driftReport?.minSamplesRequired ?? 5
  );
  const [activeTestTab, setActiveTestTab] = useState<
    'ALL' | 'KS_TEST' | 'MMD_RBF' | 'PSI_WASSERSTEIN'
  >('ALL');
  const [isExporting, setIsExporting] = useState<boolean>(false);
  const [isUpdatingConfig, setIsUpdatingConfig] = useState<boolean>(false);

  const handleExportCSV = async () => {
    setIsExporting(true);
    try {
      await api.downloadAuditCSV();
    } catch (err) {
      console.error('CSV export failed:', err);
    } finally {
      setIsExporting(false);
    }
  };

  const handleConfigChange = async (newAlpha: number, newMinSamples: number) => {
    setSelectedAlpha(newAlpha);
    setMinSamplesInput(newMinSamples);
    setIsUpdatingConfig(true);
    try {
      const updated = await api.updateMonitoringConfig({
        ksAlpha: newAlpha,
        minDriftSamples: newMinSamples,
      });
      onReportUpdated?.(updated);
    } catch (err) {
      console.error('Failed to update drift parameters:', err);
    } finally {
      setIsUpdatingConfig(false);
    }
  };

  const refCount = driftReport?.referenceSampleCount ?? driftMetrics[0]?.referenceSampleCount ?? 0;
  const currCount = driftReport?.currentSampleCount ?? inferencesCount;
  const minRequired = driftReport?.minSamplesRequired ?? minSamplesInput;
  const overallState =
    driftReport?.overallDriftState ??
    (currCount < minRequired || refCount < minRequired ? 'INSUFFICIENT_DATA' : 'HEALTHY');
  const hasInsufficientSamples = overallState === 'INSUFFICIENT_DATA';

  const ksTests = driftReport?.ksTests || [];
  const mmdTest = driftReport?.mmdTest || null;
  const distanceMetrics = driftReport?.distanceMetrics || [];
  const autoTrigger = driftReport?.autoTrigger || null;
  const mlflowStatus = driftReport?.mlflowStatus || null;

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
                <Layers className="w-4 h-4" />
              </span>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Two-Sample KS &amp; Multivariate RBF-MMD Drift Monitoring
              </h2>
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-2xl">
              Separate non-parametric <strong className="text-slate-200">Two-Sample Kolmogorov–Smirnov (KS)</strong> tests on 1D image features, <strong className="text-slate-200">Multivariate Gaussian RBF-Kernel MMD</strong> on joint 8D pixel-quality vectors, and descriptive PSI/Wasserstein distances.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={handleExportCSV}
              disabled={isExporting}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 cursor-pointer disabled:opacity-50"
            >
              <Download className="w-3.5 h-3.5 text-emerald-400" />
              <span>{isExporting ? 'Exporting...' : 'Export Audit CSV'}</span>
            </button>
            <span
              className={`px-3 py-1.5 rounded-xl border text-xs font-mono font-bold ${
                overallState === 'CRITICAL_DRIFT'
                  ? 'bg-rose-500/20 border-rose-500/40 text-rose-300'
                  : overallState === 'MODERATE_DRIFT'
                  ? 'bg-amber-500/20 border-amber-500/40 text-amber-300'
                  : overallState === 'INSUFFICIENT_DATA'
                  ? 'bg-slate-800 border-slate-700 text-amber-300'
                  : 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300'
              }`}
            >
              Status: {overallState}
            </span>
          </div>
        </div>

        {/* Sample Counts & Configurable Statistical Settings */}
        <div className="mt-4 p-3.5 rounded-xl bg-slate-950/90 border border-slate-800 text-xs text-slate-400 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-3 font-mono text-[11px]">
              <span>
                Reference N: <strong className="text-slate-200">{refCount}</strong>
              </span>
              <span className="text-slate-700">|</span>
              <span>
                Current N:{' '}
                <strong
                  className={
                    currCount < minRequired ? 'text-amber-400' : 'text-emerald-400'
                  }
                >
                  {currCount}
                </strong>
              </span>
              <span className="text-slate-700">|</span>
              <span>
                Required Minimum: <strong className="text-indigo-400">{minRequired}</strong>
              </span>
              {mlflowStatus && (
                <>
                  <span className="text-slate-700">|</span>
                  <span className="flex items-center gap-1">
                    <GitBranch className="w-3 h-3 text-indigo-400" />
                    MLflow:{' '}
                    <strong
                      className={
                        mlflowStatus.connected ? 'text-emerald-400' : 'text-slate-400'
                      }
                    >
                      {mlflowStatus.connected
                        ? 'CONNECTED'
                        : mlflowStatus.configured
                        ? 'UNREACHABLE'
                        : 'UNCONFIGURED'}
                    </strong>
                  </span>
                </>
              )}
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-1.5">
                <span className="text-[11px] text-slate-500">Min Samples:</span>
                <select
                  value={minSamplesInput}
                  disabled={isUpdatingConfig}
                  onChange={(e) =>
                    handleConfigChange(selectedAlpha, Number(e.target.value))
                  }
                  className="bg-slate-900 border border-slate-800 rounded px-2 py-0.5 text-[11px] text-indigo-400 font-mono"
                >
                  <option value={3}>N &ge; 3</option>
                  <option value={5}>N &ge; 5 (Default)</option>
                  <option value={10}>N &ge; 10</option>
                  <option value={20}>N &ge; 20</option>
                </select>
              </div>

              <div className="flex items-center gap-1.5">
                <span className="text-[11px] text-slate-500">Significance (&alpha;):</span>
                <select
                  value={selectedAlpha}
                  disabled={isUpdatingConfig}
                  onChange={(e) =>
                    handleConfigChange(Number(e.target.value), minSamplesInput)
                  }
                  className="bg-slate-900 border border-slate-800 rounded px-2 py-0.5 text-[11px] text-emerald-400 font-mono"
                >
                  <option value={0.01}>&alpha; = 0.01 (99%)</option>
                  <option value={0.05}>&alpha; = 0.05 (95%)</option>
                  <option value={0.1}>&alpha; = 0.10 (90%)</option>
                </select>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2 border-t border-slate-800/80 text-[11px]">
            <div>
              <strong className="text-slate-300">1. Two-Sample KS Test (1D):</strong>
              <code className="text-emerald-400 ml-1 font-mono">
                D = sup |F_ref(t) - F_curr(t)|
              </code>
              <p className="text-slate-500 mt-0.5">
                Tie-exact ECDF supremum with Stephens asymptotic p-value. Rejects H0 when p &lt; &alpha;.
              </p>
            </div>
            <div>
              <strong className="text-slate-300">2. Multivariate RBF-MMD (8D):</strong>
              <code className="text-indigo-400 ml-1 font-mono">
                k(x,y) = exp(-||x-y||&sup2; / 2&sigma;&sup2;)
              </code>
              <p className="text-slate-500 mt-0.5">
                Median-heuristic &sigma; on standardized 8D vectors with 200-permutation quantile threshold.
              </p>
            </div>
            <div>
              <strong className="text-slate-300">3. Descriptive Distances:</strong>
              <code className="text-amber-400 ml-1 font-mono">
                PSI &amp; Wasserstein-1
              </code>
              <p className="text-slate-500 mt-0.5">
                Reported separately from KS/MMD hypothesis tests (10-bin smoothed PSI &amp; L1 quantile distance).
              </p>
            </div>
          </div>
        </div>

        {/* Method Filter Tabs */}
        <div className="mt-4 flex flex-wrap items-center gap-2">
          {(
            [
              { id: 'ALL', label: 'All Drift Evaluations' },
              { id: 'KS_TEST', label: 'Two-Sample KS Tests (1D)' },
              { id: 'MMD_RBF', label: 'Multivariate RBF-MMD (8D Vectors)' },
              { id: 'PSI_WASSERSTEIN', label: 'PSI & Wasserstein Distances' },
            ] as const
          ).map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTestTab(tab.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                activeTestTab === tab.id
                  ? 'bg-slate-800 text-emerald-400 border border-slate-700'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Insufficient-Data State Banner */}
      {hasInsufficientSamples && (
        <div className="p-4 rounded-2xl bg-amber-950/30 border border-amber-900/50 text-amber-200 text-xs flex items-start gap-3">
          <Info className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="font-semibold text-white text-sm">
              Calculation Status: INSUFFICIENT_DATA (Reference: {refCount}, Current: {currCount}, Minimum Required: {minRequired})
            </h4>
            <p className="text-slate-300 leading-relaxed text-[11px]">
              {driftReport?.reason ||
                `Statistical KS and MMD hypothesis tests are withheld because current sample count (${currCount}) is below the required minimum (${minRequired}). No drift scores are fabricated. Run inferences in the Inference Sandbox to collect real observations.`}
            </p>
          </div>
        </div>
      )}

      {/* Automatic Drift Retraining Trigger Status */}
      {autoTrigger && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 text-xs space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2 font-semibold text-white">
              <RefreshCw className="w-3.5 h-3.5 text-emerald-400" />
              <span>End-to-End Automatic Drift Retraining Policy</span>
            </div>
            <span className="font-mono text-[11px] text-slate-400">
              Verified Labeled Pool:{' '}
              <strong
                className={
                  autoTrigger.eligibleLabeledSamplesCount >=
                    autoTrigger.minRetrainLabeledSamplesRequired &&
                  autoTrigger.eligibleDistinctClassesCount >= 2
                    ? 'text-emerald-400'
                    : 'text-amber-400'
                }
              >
                {autoTrigger.eligibleLabeledSamplesCount} /{' '}
                {autoTrigger.minRetrainLabeledSamplesRequired} samples (
                {autoTrigger.eligibleDistinctClassesCount} classes)
              </strong>
            </span>
          </div>
          {autoTrigger.lastEvent ? (
            <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-[11px] text-slate-300 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <span>
                <strong>Last Trigger Outcome ({autoTrigger.lastEvent.outcome}):</strong>{' '}
                {autoTrigger.lastEvent.reason}
              </span>
              <span className="font-mono text-slate-500 shrink-0">
                {new Date(autoTrigger.lastEvent.timestamp).toLocaleTimeString()}
              </span>
            </div>
          ) : (
            <p className="text-[11px] text-slate-400">
              Standing by: When KS (p &lt; {autoTrigger.ksAlphaThreshold}) or Multivariate RBF-MMD exceeds threshold on &ge; {autoTrigger.minDriftSamplesRequired} samples, retraining automatically enqueues if &ge; {autoTrigger.minRetrainLabeledSamplesRequired} human-reviewed labeled samples exist.
            </p>
          )}
        </div>
      )}

      {/* Section A: Multivariate RBF-Kernel MMD Test (Feature Vectors) */}
      {(activeTestTab === 'ALL' || activeTestTab === 'MMD_RBF') && mmdTest && (
        <div
          className={`p-5 rounded-2xl border space-y-4 ${
            mmdTest.status === 'SEVERE_DRIFT'
              ? 'bg-rose-950/20 border-rose-900/50'
              : mmdTest.status === 'MODERATE_DRIFT'
              ? 'bg-amber-950/20 border-amber-900/40'
              : mmdTest.status === 'INSUFFICIENT_DATA'
              ? 'bg-slate-900/60 border-slate-800/80'
              : 'bg-slate-900 border-slate-800'
          }`}
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="font-bold text-white text-sm">
                Multivariate Maximum Mean Discrepancy (MMD) — Joint 8D Feature Vector
              </h3>
              <p className="text-[11px] text-slate-400">
                Kernel: <code className="text-indigo-400">{mmdTest.kernel}</code> • Features:{' '}
                <code className="text-slate-300">{mmdTest.featureNames.join(', ')}</code>
              </p>
            </div>
            <span
              className={`px-2.5 py-1 rounded-full text-[10px] font-bold uppercase border self-start ${
                mmdTest.status === 'SEVERE_DRIFT'
                  ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                  : mmdTest.status === 'MODERATE_DRIFT'
                  ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                  : mmdTest.status === 'INSUFFICIENT_DATA'
                  ? 'bg-slate-800 text-amber-300 border-slate-700'
                  : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
              }`}
            >
              {mmdTest.status}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 text-xs">
            <div>
              <span className="text-slate-500 block text-[10px]">BIASED MMD SCORE</span>
              <span className="font-mono font-bold text-indigo-400">
                {mmdTest.mmdScore !== null ? mmdTest.mmdScore.toFixed(4) : '— (Unavailable)'}
              </span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">PERM THRESHOLD (1-&alpha;)</span>
              <span className="font-mono font-bold text-slate-300">
                {mmdTest.threshold !== null ? mmdTest.threshold.toFixed(4) : '—'}
              </span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">EMPIRICAL P-VALUE</span>
              <span
                className={`font-mono font-bold ${
                  mmdTest.pValue !== null && mmdTest.pValue <= mmdTest.alpha
                    ? 'text-rose-400'
                    : 'text-emerald-400'
                }`}
              >
                {mmdTest.pValue !== null ? mmdTest.pValue.toFixed(4) : '—'}
              </span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">BANDWIDTH (&sigma;)</span>
              <span className="font-mono font-bold text-slate-300">
                {mmdTest.bandwidthSigma !== null ? mmdTest.bandwidthSigma.toFixed(4) : '—'}
              </span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">SAMPLE COUNTS</span>
              <span className="font-mono font-bold text-slate-300">
                Ref {mmdTest.referenceSampleCount} / Curr {mmdTest.currentSampleCount}
              </span>
            </div>
          </div>

          <div className="text-[11px] text-slate-400 bg-slate-950/40 p-2.5 rounded-lg border border-slate-800/60">
            <strong>MMD Result:</strong> {mmdTest.reason}
          </div>
        </div>
      )}

      {/* Section B: Two-Sample Kolmogorov-Smirnov (KS) 1D Feature Tests */}
      {(activeTestTab === 'ALL' || activeTestTab === 'KS_TEST') && (
        <div className="space-y-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 px-1">
            Two-Sample Kolmogorov–Smirnov (KS) Tests (1D Numerical Features)
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {(ksTests.length > 0
              ? ksTests
              : driftMetrics.map((m) => ({
                  method: 'KS_TEST' as const,
                  featureName: m.featureName,
                  statistic: m.ksStatistic,
                  pValue: m.ksPValue,
                  criticalValue: m.ksCriticalValue ?? null,
                  alpha: m.alpha,
                  referenceSampleCount: m.referenceSampleCount,
                  currentSampleCount: m.currentSampleCount,
                  minSamplesRequired: m.minSamplesRequired,
                  baselineMean: m.baselineMean,
                  currentMean: m.currentMean,
                  status: m.status,
                  reason: m.reason,
                  timestamp: m.timestamp,
                }))
            ).map((ks) => {
              const isSevere = ks.status === 'SEVERE_DRIFT';
              const isModerate = ks.status === 'MODERATE_DRIFT';
              const isInsuff = ks.status === 'INSUFFICIENT_DATA' || ks.statistic === null;

              return (
                <div
                  key={ks.featureName}
                  className={`p-5 rounded-2xl border transition-all space-y-4 ${
                    isSevere
                      ? 'bg-rose-950/20 border-rose-900/50'
                      : isModerate
                      ? 'bg-amber-950/20 border-amber-900/40'
                      : isInsuff
                      ? 'bg-slate-900/60 border-slate-800/80'
                      : 'bg-slate-900 border-slate-800'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h4 className="font-semibold text-white text-sm">{ks.featureName}</h4>
                      <span className="text-[11px] text-slate-400 font-mono">
                        Ref: <strong>{ks.referenceSampleCount}</strong> • Curr:{' '}
                        <strong>{ks.currentSampleCount}</strong> • Min:{' '}
                        <strong>{ks.minSamplesRequired}</strong>
                      </span>
                    </div>

                    <span
                      className={`px-2.5 py-1 rounded-full text-[10px] font-bold uppercase border ${
                        isSevere
                          ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                          : isModerate
                          ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                          : isInsuff
                          ? 'bg-slate-800 text-amber-300 border-slate-700'
                          : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                      }`}
                    >
                      {ks.status}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 text-xs">
                    <div>
                      <span className="text-slate-500 block text-[10px]">KS STATISTIC (D)</span>
                      <span className="font-mono font-bold text-slate-200">
                        {ks.statistic !== null ? ks.statistic.toFixed(4) : '—'}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-500 block text-[10px]">CRITICAL D (&alpha;)</span>
                      <span className="font-mono font-bold text-slate-400">
                        {ks.criticalValue !== null ? ks.criticalValue.toFixed(4) : '—'}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-500 block text-[10px]">KS P-VALUE</span>
                      <span
                        className={`font-mono font-bold ${
                          ks.pValue !== null && ks.pValue < ks.alpha
                            ? 'text-rose-400'
                            : ks.pValue !== null
                            ? 'text-emerald-400'
                            : 'text-slate-500'
                        }`}
                      >
                        {ks.pValue !== null ? ks.pValue.toFixed(4) : '—'}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-500 block text-[10px]">REF / CURR MEAN</span>
                      <span className="font-mono font-bold text-slate-300">
                        {ks.baselineMean ?? '—'} / {ks.currentMean ?? '—'}
                      </span>
                    </div>
                  </div>

                  <div className="text-[11px] text-slate-400 bg-slate-950/40 p-2.5 rounded-lg border border-slate-800/60 leading-relaxed">
                    <strong>KS Decision:</strong> {ks.reason}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Section C: Separate PSI & Wasserstein-1 Descriptive Distance Metrics */}
      {(activeTestTab === 'ALL' || activeTestTab === 'PSI_WASSERSTEIN') &&
        distanceMetrics.length > 0 && (
          <div className="space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 px-1">
              Descriptive Distribution Distances (PSI &amp; Wasserstein-1 — Separate from KS/MMD)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {distanceMetrics.map((dm) => {
                const isInsuff = dm.status === 'INSUFFICIENT_DATA' || dm.psi === null;
                return (
                  <div
                    key={dm.featureName}
                    className="p-4 rounded-2xl bg-slate-900 border border-slate-800 space-y-3"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-white text-xs">{dm.featureName}</span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-400">
                        {dm.status}
                      </span>
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-xs font-mono">
                      <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                        <span className="text-[10px] text-slate-500 block">PSI (10-BIN)</span>
                        <span className="text-amber-400 font-bold">
                          {dm.psi !== null ? dm.psi.toFixed(4) : '—'}
                        </span>
                      </div>
                      <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                        <span className="text-[10px] text-slate-500 block">WASSERSTEIN-1</span>
                        <span className="text-indigo-400 font-bold">
                          {dm.wassersteinDistance !== null
                            ? dm.wassersteinDistance.toFixed(3)
                            : '—'}
                        </span>
                      </div>
                      <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                        <span className="text-[10px] text-slate-500 block">OBSERVATIONS</span>
                        <span className="text-slate-300 font-bold">
                          {dm.referenceSampleCount} / {dm.currentSampleCount}
                        </span>
                      </div>
                    </div>
                    {!isInsuff && dm.psi !== null && (
                      <div className="w-full bg-slate-950 rounded-full h-1.5 overflow-hidden border border-slate-800">
                        <div
                          className={`h-full rounded-full ${
                            dm.psi >= 0.25
                              ? 'bg-rose-500'
                              : dm.psi >= 0.1
                              ? 'bg-amber-500'
                              : 'bg-emerald-500'
                          }`}
                          style={{ width: `${Math.min(100, (dm.psi / 0.35) * 100)}%` }}
                        />
                      </div>
                    )}
                    <p className="text-[11px] text-slate-400">{dm.reason}</p>
                  </div>
                );
              })}
            </div>
          </div>
        )}
    </div>
  );
};
