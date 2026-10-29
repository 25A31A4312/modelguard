import React, { useState } from 'react';
import {
  ShieldAlert,
  CheckCircle2,
  XCircle,
  Edit3,
  Database,
  Download,
  Image as ImageIcon,
} from 'lucide-react';
import type { HumanReviewSample } from '../types/modelguard.ts';
import { api } from '../services/api.ts';

interface HumanReviewViewProps {
  queue: HumanReviewSample[];
  onQueueUpdated: (updated: HumanReviewSample) => void;
  onRetrainTriggered?: () => void;
}

export const HumanReviewView: React.FC<HumanReviewViewProps> = ({
  queue,
  onQueueUpdated,
}) => {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [overrideInput, setOverrideInput] = useState<string>('');
  const [loadedImages, setLoadedImages] = useState<Record<string, boolean>>({});
  const [brokenImages, setBrokenImages] = useState<Record<string, boolean>>({});
  const [isExporting, setIsExporting] = useState<boolean>(false);

  const retrainCandidates = queue.filter(
    (item) =>
      item.reviewed &&
      item.sendToRetraining &&
      (item.status === 'ACCEPTED' || item.status === 'OVERRIDDEN')
  );

  const handleApprove = async (id: string, currentPred: string) => {
    try {
      const updated = await api.updateHumanReview(id, {
        status: 'ACCEPTED',
        groundTruthLabel: currentPred,
        sendToRetraining: true,
      });
      onQueueUpdated(updated);
    } catch (err) {
      console.error('Failed to approve review item:', err);
    }
  };

  const handleOverrideSubmit = async (id: string) => {
    if (!overrideInput.trim()) return;
    try {
      const updated = await api.updateHumanReview(id, {
        status: 'OVERRIDDEN',
        groundTruthLabel: overrideInput.trim(),
        sendToRetraining: true,
      });
      onQueueUpdated(updated);
      setEditingId(null);
      setOverrideInput('');
    } catch (err) {
      console.error('Failed to override label:', err);
    }
  };

  const handleReject = async (id: string) => {
    try {
      const updated = await api.updateHumanReview(id, {
        status: 'REJECTED',
        sendToRetraining: false,
      });
      onQueueUpdated(updated);
    } catch (err) {
      console.error('Failed to reject review item:', err);
    }
  };

  const handleImageLoad = (id: string) => {
    setLoadedImages((prev) => ({ ...prev, [id]: true }));
  };

  const handleImageError = (id: string) => {
    setBrokenImages((prev) => ({ ...prev, [id]: true }));
    setLoadedImages((prev) => ({ ...prev, [id]: true }));
  };

  const handleExportCSV = async () => {
    setIsExporting(true);
    try {
      await api.downloadAuditCSV();
    } catch (err) {
      console.error('Failed to export CSV:', err);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/30">
                <ShieldAlert className="w-4 h-4" />
              </span>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Human-in-the-Loop Review Queue
              </h2>
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-2xl">
              Inspect original persistent inference images, verify or override ground-truth labels, and build the verified training dataset for supervised model retraining.
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
            <span className="px-3 py-1.5 rounded-full bg-slate-950 border border-slate-800 text-xs font-mono text-slate-300">
              Verified for Retraining:{' '}
              <strong className="text-emerald-400">{retrainCandidates.length}</strong>
            </span>
          </div>
        </div>
      </div>

      {/* Queue Items Feed */}
      <div className="space-y-4">
        {queue.length === 0 ? (
          <div className="p-8 text-center bg-slate-900 border border-slate-800 rounded-2xl text-xs text-slate-500">
            No items pending human review. Inferences with &lt;60% confidence, covariate drift flags, or manual sandbox dispatches appear here with their original uploaded images.
          </div>
        ) : (
          queue.map((item) => {
            const isPending = item.status === 'PENDING';
            const isEditing = editingId === item.id;
            const hasValidSrc = Boolean(item.imageUrl && item.imageUrl.trim() !== '');
            const isImageBroken = brokenImages[item.id] || !hasValidSrc;
            const isImageLoaded = loadedImages[item.id];

            return (
              <div
                key={item.id}
                className={`p-4 rounded-2xl border transition-all space-y-4 ${
                  isPending
                    ? 'bg-slate-900/90 border-slate-700/80 shadow-md'
                    : 'bg-slate-950/60 border-slate-800/80'
                }`}
              >
                <div className="flex flex-col md:flex-row items-start justify-between gap-4">
                  {/* Left: Persistent Image Preview & Association Details */}
                  <div className="flex items-start gap-4">
                    <div className="w-28 h-24 rounded-xl overflow-hidden bg-slate-950 border border-slate-700 shrink-0 relative flex items-center justify-center">
                      {isImageBroken ? (
                        <div className="flex flex-col items-center justify-center text-center p-2 text-slate-500">
                          <ImageIcon className="w-5 h-5 text-slate-600 mb-1" />
                          <span className="text-[9px] font-medium leading-tight">
                            Image Unavailable
                          </span>
                        </div>
                      ) : (
                        <>
                          {!isImageLoaded && (
                            <div className="absolute inset-0 bg-slate-900 animate-pulse flex items-center justify-center">
                              <span className="text-[9px] text-slate-500 font-mono">Loading...</span>
                            </div>
                          )}
                          <img
                            src={item.imageUrl}
                            alt={`Review ${item.id}`}
                            onLoad={() => handleImageLoad(item.id)}
                            onError={() => handleImageError(item.id)}
                            className="w-full h-full object-cover"
                          />
                        </>
                      )}
                      <span className="absolute bottom-1 right-1 px-1 py-0.5 rounded text-[8px] font-mono bg-slate-950/85 text-slate-300">
                        {item.detectedCount} obj
                      </span>
                    </div>

                    <div className="space-y-1.5">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-xs text-slate-400 font-mono">
                          Review: {item.id}
                        </span>
                        {item.inferenceId && (
                          <>
                            <span className="text-slate-600">•</span>
                            <span className="text-[11px] text-indigo-300 font-mono">
                              Inference: {item.inferenceId}
                            </span>
                          </>
                        )}
                        <span className="text-slate-600">•</span>
                        <span className="text-xs text-slate-500">
                          {new Date(item.timestamp).toLocaleTimeString()}
                        </span>
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            item.status === 'PENDING'
                              ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                              : item.status === 'ACCEPTED'
                              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                              : item.status === 'OVERRIDDEN'
                              ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                              : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                          }`}
                        >
                          {item.status}
                        </span>
                      </div>

                      <div className="text-sm font-semibold text-white">
                        Model Prediction:{' '}
                        <span className="text-slate-200 capitalize">
                          {item.modelPrediction}
                        </span>{' '}
                        <span className="text-xs text-amber-400 font-mono">
                          ({(item.modelConfidence * 100).toFixed(1)}% conf)
                        </span>
                      </div>

                      <div className="text-xs text-slate-400">
                        Flagged Reason:{' '}
                        <span className="text-slate-300">{item.driftReason}</span>
                      </div>

                      {item.groundTruthLabel && (
                        <div className="text-xs text-emerald-400 font-medium">
                          Verified Ground Truth: <strong>{item.groundTruthLabel}</strong>
                          {item.reviewedBy && (
                            <span className="text-slate-400 font-normal ml-2">
                              (by {item.reviewedBy})
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Right: Operator Actions */}
                  <div className="flex flex-col sm:flex-row items-end sm:items-center gap-2 self-end md:self-center shrink-0">
                    {isPending ? (
                      <>
                        <button
                          onClick={() => handleApprove(item.id, item.modelPrediction)}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/30 text-xs font-semibold cursor-pointer"
                        >
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          <span>Approve Label</span>
                        </button>

                        <button
                          onClick={() => {
                            setEditingId(item.id);
                            setOverrideInput(item.modelPrediction);
                          }}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-300 border border-indigo-500/30 text-xs font-semibold cursor-pointer"
                        >
                          <Edit3 className="w-3.5 h-3.5" />
                          <span>Correct Label</span>
                        </button>

                        <button
                          onClick={() => handleReject(item.id)}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 text-xs cursor-pointer"
                        >
                          <XCircle className="w-3.5 h-3.5" />
                          <span>Reject</span>
                        </button>
                      </>
                    ) : (
                      <div className="flex items-center gap-2">
                        {item.sendToRetraining && (
                          <span className="flex items-center gap-1 text-xs text-emerald-400 font-medium bg-emerald-950/40 px-2.5 py-1 rounded-lg border border-emerald-800/40">
                            <Database className="w-3.5 h-3.5" />
                            Eligible for Retraining
                          </span>
                        )}
                        <button
                          onClick={() => {
                            setEditingId(item.id);
                            setOverrideInput(item.groundTruthLabel || item.modelPrediction);
                          }}
                          className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs cursor-pointer"
                        >
                          Edit Label
                        </button>
                      </div>
                    )}
                  </div>
                </div>

                {/* Inline Editing Form */}
                {isEditing && (
                  <div className="p-3 bg-slate-950 rounded-xl border border-indigo-900/50 space-y-2">
                    <label className="text-xs text-indigo-300 font-medium block">
                      Assign True Ground Truth Label:
                    </label>
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={overrideInput}
                        onChange={(e) => setOverrideInput(e.target.value)}
                        placeholder="e.g. sports car, foggy roadway vehicle, truck..."
                        className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500 font-medium"
                      />
                      <button
                        onClick={() => handleOverrideSubmit(item.id)}
                        className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold cursor-pointer"
                      >
                        Save Ground Truth
                      </button>
                      <button
                        onClick={() => setEditingId(null)}
                        className="px-3 py-1.5 bg-slate-800 text-slate-400 rounded-lg text-xs cursor-pointer"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
