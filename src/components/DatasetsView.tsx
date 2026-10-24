import React from 'react';
import { Database } from 'lucide-react';
import type { ReferenceDataset } from '../types/modelguard.ts';

interface DatasetsViewProps {
  datasets: ReferenceDataset[];
}

export const DatasetsView: React.FC<DatasetsViewProps> = ({ datasets }) => {
  return (
    <div className="space-y-6">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                <Database className="w-4 h-4" />
              </span>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Reference Baseline Datasets
              </h2>
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-2xl">
              Genuine raw reference observations computed reproducibly by decoding uncompressed RGBA spatial sub-patches from verified baseline photographs (no Gaussian synthesis).
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="px-3 py-1 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono text-slate-300">
              Active Baselines: <strong className="text-white">{datasets.length}</strong>
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {datasets.map((dataset, i) => (
          <div
            key={dataset.id}
            className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-4 shadow-sm"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <h3 className="font-bold text-white text-base">{dataset.name}</h3>
                <span className="text-xs text-slate-400">{dataset.domain}</span>
              </div>
              <span
                className={`px-2.5 py-1 rounded-full text-[10px] font-bold border ${
                  i === 0
                    ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                    : 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30'
                }`}
              >
                {i === 0 ? 'PRIMARY REFERENCE' : 'SECONDARY REFERENCE'}
              </span>
            </div>

            {dataset.sourceDescription && (
              <p className="text-[11px] text-slate-400 bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
                {dataset.sourceDescription}
              </p>
            )}

            <div className="grid grid-cols-4 gap-2 p-3 bg-slate-950 rounded-xl border border-slate-800/80 text-xs">
              <div>
                <span className="text-slate-500 block text-[10px]">OBSERVATIONS</span>
                <span className="font-mono font-bold text-slate-200">
                  {dataset.sampleCount.toLocaleString()}
                </span>
              </div>
              <div>
                <span className="text-slate-500 block text-[10px]">MEAN SHARPNESS</span>
                <span className="font-mono font-bold text-emerald-400">
                  {dataset.meanSharpness}
                </span>
              </div>
              <div>
                <span className="text-slate-500 block text-[10px]">MEAN BRIGHTNESS</span>
                <span className="font-mono font-bold text-slate-200">
                  {dataset.meanBrightness}
                </span>
              </div>
              <div>
                <span className="text-slate-500 block text-[10px]">MEAN CONTRAST</span>
                <span className="font-mono font-bold text-indigo-400">
                  {dataset.meanContrast}
                </span>
              </div>
            </div>

            <div className="space-y-1.5">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                Common Reference Classes
              </span>
              <div className="flex flex-wrap gap-1.5">
                {dataset.commonClasses.map((cls, idx) => (
                  <span
                    key={idx}
                    className="px-2 py-0.5 rounded text-[11px] bg-slate-800 text-slate-300 border border-slate-700 capitalize"
                  >
                    {cls}
                  </span>
                ))}
              </div>
            </div>

            <div className="text-[11px] text-slate-500 flex items-center justify-between pt-2 border-t border-slate-800/80">
              <span>ID: {dataset.id}</span>
              <span>Updated: {new Date(dataset.lastUpdated).toLocaleDateString()}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
