import React, { useState } from 'react';
import {
  RefreshCw,
  Play,
  GitBranch,
  Download,
  ShieldCheck,
  AlertCircle,
  Database,
} from 'lucide-react';
import type {
  DriftAutoTriggerStatus,
  RetrainingJob,
} from '../types/modelguard.ts';
import { api } from '../services/api.ts';

interface RetrainingViewProps {
  jobs: RetrainingJob[];
  activeJob: RetrainingJob | null;
  eligibility?: {
    eligibleLabeledSamplesCount: number;
    eligibleDistinctClassesCount: number;
    minRequiredSamples: number;
    readyToTrain: boolean;
  } | null;
  autoTrigger?: DriftAutoTriggerStatus | null;
  pendingReviewCount: number;
  onJobStarted: (job: RetrainingJob) => void;
}

export const RetrainingView: React.FC<RetrainingViewProps> = ({
  jobs,
  activeJob,
  eligibility,
  autoTrigger,
  onJobStarted,
}) => {
  const [isStarting, setIsStarting] = useState(false);
  const [triggerSource, setTriggerSource] = useState<
    'MANUAL' | 'HUMAN_REVIEW_BATCH' | 'AUTO_DRIFT_THRESHOLD'
  >('HUMAN_REVIEW_BATCH');
  const [isExporting, setIsExporting] = useState(false);
  const [triggerError, setTriggerError] = useState<string | null>(null);

  const handleStartRetraining = async () => {
    setIsStarting(true);
    setTriggerError(null);
    try {
      const job = await api.triggerRetrain(triggerSource);
      onJobStarted(job);
      if (job.status === 'FAILED' && job.errorMessage) {
        setTriggerError(job.errorMessage);
      }
    } catch (err: any) {
      setTriggerError(err.message || 'Failed to execute retraining job');
    } finally {
      setIsStarting(false);
    }
  };

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

  const eligibleCount =
    eligibility?.eligibleLabeledSamplesCount ??
    autoTrigger?.eligibleLabeledSamplesCount ??
    0;
  const distinctClasses =
    eligibility?.eligibleDistinctClassesCount ??
    autoTrigger?.eligibleDistinctClassesCount ??
    0;
  const minRequired =
    eligibility?.minRequiredSamples ??
    autoTrigger?.minRetrainLabeledSamplesRequired ??
    6;
  const readyToTrain = eligibleCount >= minRequired && distinctClasses >= 2;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                <RefreshCw className="w-4 h-4" />
              </span>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Supervised Retraining &amp; Validation Pipeline
              </h2>
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-2xl">
              Trains an L2-regularized Multinomial Softmax Logistic Regression classifier on genuine 8D pixel-feature vectors from human-reviewed samples, evaluates on a stratified held-out validation split, and blocks promotion unless validation criteria pass.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleExportCSV}
              disabled={isExporting}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 cursor-pointer disabled:opacity-50"
            >
              <Download className="w-3.5 h-3.5 text-emerald-400" />
              <span>{isExporting ? 'Exporting...' : 'Export Audit CSV'}</span>
            </button>
            <button
              onClick={handleStartRetraining}
              disabled={isStarting || Boolean(activeJob)}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-600 text-slate-950 text-xs font-semibold shadow-lg shadow-emerald-500/20 transition-all cursor-pointer disabled:opacity-50"
            >
              <Play className="w-4 h-4 fill-current" />
              <span>
                {isStarting || activeJob ? 'Training in Progress...' : 'Run Retraining Job'}
              </span>
            </button>
          </div>
        </div>

        {/* Dataset Readiness & Trigger Source */}
        <div className="mt-5 pt-4 border-t border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-slate-400 font-medium">Trigger Source:</span>
            {(
              [
                { id: 'HUMAN_REVIEW_BATCH', label: 'Human Review Batch' },
                { id: 'AUTO_DRIFT_THRESHOLD', label: 'Drift Threshold Policy' },
                { id: 'MANUAL', label: 'Manual Execution' },
              ] as const
            ).map((src) => (
              <button
                key={src.id}
                onClick={() => setTriggerSource(src.id)}
                className={`px-3 py-1.5 rounded-lg font-semibold transition-all cursor-pointer ${
                  triggerSource === src.id
                    ? 'bg-slate-800 text-emerald-400 border border-slate-700'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {src.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2 font-mono text-[11px] bg-slate-950 px-3 py-1.5 rounded-xl border border-slate-800">
            <Database className="w-3.5 h-3.5 text-indigo-400" />
            <span>
              Eligible Reviewed Data:{' '}
              <strong className={readyToTrain ? 'text-emerald-400' : 'text-amber-400'}>
                {eligibleCount} / {minRequired} min
              </strong>{' '}
              ({distinctClasses} / 2 min classes)
            </span>
          </div>
        </div>
      </div>

      {/* Readiness Warning / Error Banner */}
      {!readyToTrain && (
        <div className="p-4 rounded-2xl bg-amber-950/30 border border-amber-900/50 text-amber-200 text-xs flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="font-semibold text-white text-sm">
              Insufficient Human-Reviewed Labeled Data ({eligibleCount}/{minRequired} samples, {distinctClasses}/2 classes)
            </h4>
            <p className="text-slate-300 text-[11px] leading-relaxed">
              ModelGuard never fabricates training samples or fake loss curves. Approve or correct at least <strong>{minRequired} samples across &ge; 2 ground-truth classes</strong> in the <strong>Human Review</strong> tab before retraining can produce a validated candidate checkpoint.
            </p>
          </div>
        </div>
      )}

      {triggerError && (
        <div className="p-4 rounded-2xl bg-rose-950/30 border border-rose-900/50 text-rose-200 text-xs flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <strong className="block text-white mb-0.5">Retraining Execution Report</strong>
            <span>{triggerError}</span>
          </div>
        </div>
      )}

      {/* Retraining Run History */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4">
        <h3 className="font-semibold text-white text-sm">Retraining Run History</h3>
        {jobs.length === 0 ? (
          <div className="p-8 text-center bg-slate-950/50 border border-slate-800 rounded-xl text-xs text-slate-500">
            No retraining jobs executed yet. Only genuine training attempts and automatic drift-triggered runs are recorded here.
          </div>
        ) : (
          <div className="space-y-3">
            {jobs.map((job) => {
              const isFailed = job.status === 'FAILED';
              const passedGate = job.validationReport?.passedPromotionCriteria === true;

              return (
                <div
                  key={job.id}
                  className={`p-4 rounded-xl border flex flex-col md:flex-row items-start md:items-center justify-between gap-4 ${
                    isFailed
                      ? 'bg-rose-950/15 border-rose-900/40'
                      : 'bg-slate-950/70 border-slate-800/80'
                  }`}
                >
                  <div className="space-y-1.5 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono font-bold text-sm text-white">{job.id}</span>
                      <span className="font-mono text-xs text-indigo-400 bg-indigo-950/40 px-2 py-0.5 rounded border border-indigo-900/40">
                        {job.candidateVersion}
                      </span>
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          job.status === 'COMPLETED'
                            ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                            : isFailed
                            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                            : 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                        }`}
                      >
                        {job.status}
                      </span>
                      {job.durationMs != null && (
                        <span className="text-[11px] font-mono text-slate-500">
                          ({job.durationMs}ms)
                        </span>
                      )}
                    </div>

                    <div className="text-xs text-slate-400 flex flex-wrap items-center gap-3">
                      <span>Trigger: {job.triggerSource}</span>
                      <span>•</span>
                      <span>
                        Samples: {job.newSamplesCount} eligible (Train: {job.trainSamplesCount ?? 0}, Val:{' '}
                        {job.valSamplesCount ?? 0})
                      </span>
                      <span>•</span>
                      <span>Started: {new Date(job.startedAt).toLocaleString()}</span>
                      {job.mlflowRunId && (
                        <span className="font-mono text-indigo-400 flex items-center gap-1">
                          <GitBranch className="w-3 h-3" />
                          MLflow: {job.mlflowRunId}
                        </span>
                      )}
                    </div>

                    {job.errorMessage && (
                      <div className="mt-1.5 text-[11px] text-rose-300 bg-rose-950/40 px-2.5 py-1.5 rounded border border-rose-900/50">
                        {job.errorMessage}
                      </div>
                    )}

                    {job.validationReport && (
                      <div
                        className={`mt-2 text-[11px] px-2.5 py-1.5 rounded border flex items-start gap-2 ${
                          passedGate
                            ? 'text-emerald-200 bg-emerald-950/30 border-emerald-800/50'
                            : 'text-amber-200 bg-amber-950/30 border-amber-800/50'
                        }`}
                      >
                        <ShieldCheck
                          className={`w-4 h-4 shrink-0 mt-0.5 ${
                            passedGate ? 'text-emerald-400' : 'text-amber-400'
                          }`}
                        />
                        <span>{job.validationReport.promotionPolicyReason}</span>
                      </div>
                    )}
                  </div>

                  <div className="flex items-center gap-4 text-xs font-mono shrink-0">
                    <div>
                      <span className="text-slate-500 block text-[10px]">TRAIN / VAL LOSS</span>
                      <span className="text-slate-200">
                        {job.trainLoss !== null ? job.trainLoss.toFixed(4) : '—'} /{' '}
                        {job.valLoss !== null ? job.valLoss.toFixed(4) : '—'}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-500 block text-[10px]">ACC DELTA</span>
                      <span
                        className={`font-bold ${
                          job.accuracyDelta !== null && job.accuracyDelta >= 0
                            ? 'text-emerald-400'
                            : 'text-slate-400'
                        }`}
                      >
                        {job.accuracyDelta !== null
                          ? `${job.accuracyDelta >= 0 ? '+' : ''}${(
                              job.accuracyDelta * 100
                            ).toFixed(1)}%`
                          : '—'}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
