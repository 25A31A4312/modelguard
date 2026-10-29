import React, { useState } from 'react';
import {
  AlertTriangle,
  Layers,
  TrendingDown,
  CheckCircle2,
  Clock,
  Filter,
  ShieldAlert,
  ArrowRight,
  Sliders,
  Download,
} from 'lucide-react';
import type { AlertItem, AlertSeverity, AlertStatus, AlertType } from '../types/modelguard.ts';
import { api } from '../services/api.ts';

interface AlertsViewProps {
  alerts: AlertItem[];
  onAlertUpdated: (updated: AlertItem) => void;
}

export const AlertsView: React.FC<AlertsViewProps> = ({ alerts, onAlertUpdated }) => {
  const [selectedType, setSelectedType] = useState<string>('ALL');
  const [selectedStatus, setSelectedStatus] = useState<string>('ALL');

  const filteredAlerts = alerts.filter((alert) => {
    if (selectedType !== 'ALL' && alert.type !== selectedType) return false;
    if (selectedStatus !== 'ALL' && alert.status !== selectedStatus) return false;
    return true;
  });

  const handleStatusChange = async (alertId: string, newStatus: AlertStatus) => {
    try {
      const updated = await api.updateAlertStatus(alertId, newStatus);
      onAlertUpdated(updated);
    } catch (err) {
      console.error('Failed to update alert status:', err);
    }
  };

  const dataDriftCount = alerts.filter((a) => a.type === 'DATA_DRIFT' && a.status === 'OPEN').length;
  const lowConfCount = alerts.filter((a) => a.type === 'LOW_CONFIDENCE' && a.status === 'OPEN').length;
  const openTotal = alerts.filter((a) => a.status === 'OPEN').length;

  return (
    <div className="space-y-6">
      {/* Header and Separation Notice */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/30">
                <AlertTriangle className="w-4 h-4" />
              </span>
              <h2 className="text-lg font-bold text-white tracking-tight">
                MLOps Reliability &amp; Drift Alert Center
              </h2>
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-2xl">
              Strict separation between <strong className="text-slate-200">Data Drift Alerts</strong> (input distribution shifts such as fog or brightness) and <strong className="text-slate-200">Low-Confidence Alerts</strong> (model uncertainty on edge cases).
            </p>
          </div>

          {/* Quick counts & Export */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => api.downloadAuditCSV()}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition-all cursor-pointer shadow-sm"
            >
              <Download className="w-3.5 h-3.5 text-emerald-400" />
              <span>Export Audit CSV</span>
            </button>
            <span className="px-3 py-1.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs font-semibold">
              {openTotal} Open Alerts
            </span>
          </div>
        </div>

        {/* Category Tabs: Separate Data Drift vs Low Confidence */}
        <div className="mt-5 flex flex-wrap items-center gap-2 border-t border-slate-800 pt-4">
          <button
            onClick={() => setSelectedType('ALL')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              selectedType === 'ALL'
                ? 'bg-slate-800 text-white border border-slate-700 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            All Alerts ({alerts.length})
          </button>
          <button
            onClick={() => setSelectedType('DATA_DRIFT')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 cursor-pointer ${
              selectedType === 'DATA_DRIFT'
                ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-sm'
                : 'text-slate-400 hover:text-rose-400'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Data Drift Alerts ({dataDriftCount} open)</span>
          </button>
          <button
            onClick={() => setSelectedType('LOW_CONFIDENCE')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 cursor-pointer ${
              selectedType === 'LOW_CONFIDENCE'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm'
                : 'text-slate-400 hover:text-amber-400'
            }`}
          >
            <TrendingDown className="w-3.5 h-3.5" />
            <span>Low Confidence Alerts ({lowConfCount} open)</span>
          </button>

          {/* Status Filter */}
          <div className="ml-auto flex items-center gap-2 text-xs">
            <span className="text-slate-500">Status:</span>
            <select
              value={selectedStatus}
              onChange={(e) => setSelectedStatus(e.target.value)}
              className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-slate-300 text-xs focus:outline-none focus:border-slate-700 cursor-pointer"
            >
              <option value="ALL">All Statuses</option>
              <option value="OPEN">Open Only</option>
              <option value="INVESTIGATING">Investigating</option>
              <option value="RESOLVED">Resolved</option>
            </select>
          </div>
        </div>
      </div>

      {/* Alerts Feed */}
      <div className="space-y-3">
        {filteredAlerts.length === 0 ? (
          <div className="p-8 text-center bg-slate-900 border border-slate-800 rounded-2xl text-xs text-slate-500">
            No alerts found matching the current filter criteria.
          </div>
        ) : (
          filteredAlerts.map((alert) => {
            const isDrift = alert.type === 'DATA_DRIFT';
            const isOpen = alert.status === 'OPEN';
            const isInvestigating = alert.status === 'INVESTIGATING';
            const isResolved = alert.status === 'RESOLVED';

            return (
              <div
                key={alert.id}
                className={`p-4 rounded-xl border transition-all space-y-3 ${
                  isOpen
                    ? isDrift
                      ? 'bg-rose-950/20 border-rose-900/40'
                      : 'bg-amber-950/20 border-amber-900/40'
                    : 'bg-slate-900/80 border-slate-800'
                }`}
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold tracking-wider uppercase border ${
                        alert.severity === 'CRITICAL'
                          ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                          : alert.severity === 'HIGH'
                          ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                          : 'bg-slate-800 text-slate-300 border-slate-700'
                      }`}
                    >
                      {alert.severity}
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                        isDrift
                          ? 'bg-purple-500/10 text-purple-300 border-purple-500/30'
                          : 'bg-amber-500/10 text-amber-300 border-amber-500/30'
                      }`}
                    >
                      {alert.type.replace('_', ' ')}
                    </span>
                    <span className="text-xs text-slate-500 font-mono">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </span>
                  </div>

                  {/* Status Indicator */}
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                        isOpen
                          ? 'text-rose-400 bg-rose-950/40 border border-rose-900/50'
                          : isInvestigating
                          ? 'text-amber-400 bg-amber-950/40 border border-amber-900/50'
                          : 'text-emerald-400 bg-emerald-950/40 border border-emerald-900/50'
                      }`}
                    >
                      {alert.status}
                    </span>
                  </div>
                </div>

                <div>
                  <h3 className="text-sm font-semibold text-white">{alert.title}</h3>
                  <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                    {alert.description}
                  </p>
                </div>

                {/* Suggested Action & Triage Buttons */}
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pt-2 border-t border-slate-800/80 text-xs">
                  <div className="text-[11px] text-slate-400 flex items-center gap-1.5">
                    <strong className="text-slate-300">Suggested Action:</strong>
                    <span>{alert.suggestedAction}</span>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    {isOpen && (
                      <button
                        onClick={() => handleStatusChange(alert.id, 'INVESTIGATING')}
                        className="px-2.5 py-1 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/30 text-[11px] font-semibold cursor-pointer"
                      >
                        Start Investigation
                      </button>
                    )}
                    {(isOpen || isInvestigating) && (
                      <button
                        onClick={() => handleStatusChange(alert.id, 'RESOLVED')}
                        className="px-2.5 py-1 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/30 text-[11px] font-semibold cursor-pointer"
                      >
                        Resolve Alert
                      </button>
                    )}
                    {!isResolved && (
                      <button
                        onClick={() => handleStatusChange(alert.id, 'DISMISSED')}
                        className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 text-[11px] cursor-pointer"
                      >
                        Dismiss
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
