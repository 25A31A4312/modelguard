import React from 'react';
import {
  ShieldAlert,
  Activity,
  Cpu,
  Layers,
  GitBranch,
  RefreshCw,
  Database,
  Eye,
  Server,
  AlertTriangle,
} from 'lucide-react';

export type ActiveTab =
  | 'dashboard'
  | 'inference'
  | 'drift'
  | 'alerts'
  | 'review'
  | 'registry'
  | 'retraining'
  | 'datasets';

interface NavbarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  openAlertsCount: number;
  pendingReviewCount: number;
  serverOnline: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  openAlertsCount,
  pendingReviewCount,
  serverOnline,
}) => {
  const navItems = [
    { id: 'dashboard' as ActiveTab, label: 'Dashboard', icon: Activity },
    { id: 'inference' as ActiveTab, label: 'Inference Sandbox', icon: Eye },
    { id: 'drift' as ActiveTab, label: 'Drift Monitoring', icon: Layers },
    {
      id: 'alerts' as ActiveTab,
      label: 'Alerts',
      icon: AlertTriangle,
      badge: openAlertsCount > 0 ? openAlertsCount : null,
      badgeColor: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
    },
    {
      id: 'review' as ActiveTab,
      label: 'Human Review',
      icon: ShieldAlert,
      badge: pendingReviewCount > 0 ? pendingReviewCount : null,
      badgeColor: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
    },
    { id: 'registry' as ActiveTab, label: 'Model Registry', icon: Cpu },
    { id: 'retraining' as ActiveTab, label: 'Retraining', icon: RefreshCw },
    { id: 'datasets' as ActiveTab, label: 'Datasets', icon: Database },
  ];

  return (
    <header className="sticky top-0 z-50 border-b border-slate-800 bg-slate-950/90 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo & Name */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-emerald-500 p-0.5 shadow-lg shadow-indigo-500/20">
              <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <ShieldAlert className="w-5 h-5 text-emerald-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg tracking-tight text-white font-mono">
                  Model<span className="text-emerald-400">Guard</span>
                </span>
                <span className="px-2 py-0.5 text-[10px] font-semibold tracking-wider uppercase rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
                  CV MLOps
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Computer Vision Drift &amp; Reliability Platform
              </p>
            </div>
          </div>

          {/* Operational Status Badges */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs">
              <div
                className={`w-2 h-2 rounded-full ${
                  serverOnline ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'
                }`}
              />
              <span className="text-slate-300 font-mono">PORT 3000</span>
              <span className="text-slate-600">|</span>
              <span className="text-emerald-400 font-medium">Cloud Run Ready</span>
            </div>

            <div className="hidden md:flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-900/60 border border-slate-800 text-xs text-slate-400">
              <Server className="w-3.5 h-3.5 text-indigo-400" />
              <span>0.0.0.0 Binding Verified</span>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="flex space-x-1 overflow-x-auto pb-2 pt-1 no-scrollbar border-t border-slate-900">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all whitespace-nowrap cursor-pointer ${
                  isActive
                    ? 'bg-slate-800 text-emerald-400 shadow-sm border border-slate-700'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-emerald-400' : 'text-slate-400'}`} />
                <span>{item.label}</span>
                {item.badge !== null && (
                  <span
                    className={`ml-1 px-1.5 py-0.2 rounded-full text-[10px] font-bold border ${item.badgeColor}`}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
