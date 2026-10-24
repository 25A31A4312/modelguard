/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Navbar, ActiveTab } from './components/Navbar.tsx';
import { DashboardView } from './components/DashboardView.tsx';
import { InferenceSandbox } from './components/InferenceSandbox.tsx';
import { DriftMonitoringView } from './components/DriftMonitoringView.tsx';
import { AlertsView } from './components/AlertsView.tsx';
import { HumanReviewView } from './components/HumanReviewView.tsx';
import { ModelRegistryView } from './components/ModelRegistryView.tsx';
import { RetrainingView } from './components/RetrainingView.tsx';
import { DatasetsView } from './components/DatasetsView.tsx';
import { api } from './services/api.ts';
import type {
  AlertItem,
  DriftEvaluationReport,
  DriftMetric,
  HumanReviewSample,
  InferenceResult,
  ModelVersion,
  ReferenceDataset,
  RetrainingJob,
} from './types/modelguard.ts';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('dashboard');
  const [serverOnline, setServerOnline] = useState<boolean>(true);
  const [inferences, setInferences] = useState<InferenceResult[]>([]);
  const [driftReport, setDriftReport] = useState<DriftEvaluationReport | null>(null);
  const [driftMetrics, setDriftMetrics] = useState<DriftMetric[]>([]);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [reviewQueue, setReviewQueue] = useState<HumanReviewSample[]>([]);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [measuredPerf, setMeasuredPerf] = useState<any>(null);
  const [retrainingJobs, setRetrainingJobs] = useState<RetrainingJob[]>([]);
  const [activeRetrainJob, setActiveRetrainJob] = useState<RetrainingJob | null>(null);
  const [retrainEligibility, setRetrainEligibility] = useState<{
    eligibleLabeledSamplesCount: number;
    eligibleDistinctClassesCount: number;
    minRequiredSamples: number;
    readyToTrain: boolean;
  } | null>(null);
  const [referenceDatasets, setReferenceDatasets] = useState<ReferenceDataset[]>([]);

  const fetchAllData = useCallback(async () => {
    try {
      await api.checkHealth();
      setServerOnline(true);
    } catch {
      setServerOnline(false);
    }

    try {
      const [
        infRes,
        driftRes,
        alertsRes,
        reviewRes,
        modelsRes,
        retrainRes,
        datasetsRes,
      ] = await Promise.all([
        api.getInferences(),
        api.getDriftMetrics(),
        api.getAlerts(),
        api.getHumanReviewQueue(),
        api.getModels(),
        api.getRetrainingJobs(),
        api.getReferenceDatasets(),
      ]);

      setInferences(infRes.inferences || []);
      setDriftReport(driftRes);
      setDriftMetrics(driftRes.metrics || []);
      setAlerts(alertsRes.alerts || []);
      setReviewQueue(reviewRes.queue || []);
      setModels(modelsRes.models || []);
      setMeasuredPerf(modelsRes.measuredPerformance || null);
      setRetrainingJobs(retrainRes.jobs || []);
      setActiveRetrainJob(retrainRes.activeJob || null);
      if (retrainRes.eligibility) setRetrainEligibility(retrainRes.eligibility);
      setReferenceDatasets(datasetsRes.datasets || []);
    } catch (err) {
      console.error('Failed to load initial data:', err);
    }
  }, []);

  useEffect(() => {
    fetchAllData();
  }, [fetchAllData]);

  // Poll retraining status when a job is active
  useEffect(() => {
    if (!activeRetrainJob) return;
    const timer = setInterval(async () => {
      try {
        const retrainRes = await api.getRetrainingJobs();
        setRetrainingJobs(retrainRes.jobs || []);
        setActiveRetrainJob(retrainRes.activeJob || null);
        if (retrainRes.eligibility) setRetrainEligibility(retrainRes.eligibility);

        if (!retrainRes.activeJob) {
          const modelsRes = await api.getModels();
          setModels(modelsRes.models || []);
        }
      } catch {
        // ignore transient polling error
      }
    }, 1500);

    return () => clearInterval(timer);
  }, [activeRetrainJob]);

  const handleInferenceCompleted = async (newResult: InferenceResult) => {
    setInferences((prev) => [newResult, ...prev.filter((i) => i.id !== newResult.id)]);

    try {
      const [driftRes, alertsRes, reviewRes, retrainRes, modelsRes] = await Promise.all([
        api.getDriftMetrics(),
        api.getAlerts(),
        api.getHumanReviewQueue(),
        api.getRetrainingJobs(),
        api.getModels(),
      ]);
      setDriftReport(driftRes);
      setDriftMetrics(driftRes.metrics || []);
      setAlerts(alertsRes.alerts || []);
      setReviewQueue(reviewRes.queue || []);
      setRetrainingJobs(retrainRes.jobs || []);
      setActiveRetrainJob(retrainRes.activeJob || null);
      if (retrainRes.eligibility) setRetrainEligibility(retrainRes.eligibility);
      setModels(modelsRes.models || []);
      setMeasuredPerf(modelsRes.measuredPerformance || null);
    } catch (err) {
      console.error('Error refreshing post-inference metrics:', err);
    }
  };

  const handleAlertUpdated = (updatedAlert: AlertItem) => {
    setAlerts((prev) => prev.map((a) => (a.id === updatedAlert.id ? updatedAlert : a)));
  };

  const handleReviewUpdated = async (updatedItem: HumanReviewSample) => {
    setReviewQueue((prev) => {
      const exists = prev.some((q) => q.id === updatedItem.id);
      return exists
        ? prev.map((q) => (q.id === updatedItem.id ? updatedItem : q))
        : [updatedItem, ...prev];
    });
    try {
      const [retrainRes, modelsRes, driftRes] = await Promise.all([
        api.getRetrainingJobs(),
        api.getModels(),
        api.getDriftMetrics(),
      ]);
      if (retrainRes.eligibility) setRetrainEligibility(retrainRes.eligibility);
      setMeasuredPerf(modelsRes.measuredPerformance || null);
      setDriftReport(driftRes);
    } catch {
      // ignore
    }
  };

  const handleJobStarted = async (newJob: RetrainingJob) => {
    setRetrainingJobs((prev) => [newJob, ...prev.filter((j) => j.id !== newJob.id)]);
    if (
      newJob.status === 'QUEUED' ||
      newJob.status === 'TRAINING' ||
      newJob.status === 'VALIDATING'
    ) {
      setActiveRetrainJob(newJob);
    } else {
      setActiveRetrainJob(null);
      const modelsRes = await api.getModels();
      setModels(modelsRes.models || []);
    }
  };

  const openAlertsCount = alerts.filter((a) => a.status === 'OPEN').length;
  const pendingReviewCount = reviewQueue.filter((q) => q.status === 'PENDING').length;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-emerald-500/30 selection:text-emerald-300">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        openAlertsCount={openAlertsCount}
        pendingReviewCount={pendingReviewCount}
        serverOnline={serverOnline}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'dashboard' && (
          <DashboardView
            inferences={inferences}
            driftMetrics={driftMetrics}
            driftReport={driftReport}
            alerts={alerts}
            models={models}
            measuredPerformance={measuredPerf}
            pendingReviewCount={pendingReviewCount}
            setActiveTab={setActiveTab}
          />
        )}

        {activeTab === 'inference' && (
          <InferenceSandbox
            onInferenceCompleted={handleInferenceCompleted}
            onReviewEnqueued={handleReviewUpdated}
          />
        )}

        {activeTab === 'drift' && (
          <DriftMonitoringView
            driftMetrics={driftMetrics}
            driftReport={driftReport}
            inferencesCount={driftReport?.currentSampleCount ?? inferences.length}
            onReportUpdated={(rep) => {
              setDriftReport(rep);
              setDriftMetrics(rep.metrics || []);
            }}
          />
        )}

        {activeTab === 'alerts' && (
          <AlertsView alerts={alerts} onAlertUpdated={handleAlertUpdated} />
        )}

        {activeTab === 'review' && (
          <HumanReviewView
            queue={reviewQueue}
            onQueueUpdated={handleReviewUpdated}
            onRetrainTriggered={() => setActiveTab('retraining')}
          />
        )}

        {activeTab === 'registry' && (
          <ModelRegistryView models={models} onModelPromoted={fetchAllData} />
        )}

        {activeTab === 'retraining' && (
          <RetrainingView
            jobs={retrainingJobs}
            activeJob={activeRetrainJob}
            eligibility={retrainEligibility}
            autoTrigger={driftReport?.autoTrigger || null}
            pendingReviewCount={pendingReviewCount}
            onJobStarted={handleJobStarted}
          />
        )}

        {activeTab === 'datasets' && <DatasetsView datasets={referenceDatasets} />}
      </main>

      <footer className="border-t border-slate-900 bg-slate-950/80 py-4 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>ModelGuard — Computer Vision Drift Monitoring &amp; MLOps Platform</span>
          <span className="font-mono text-[11px] text-slate-600">
            Host: 0.0.0.0 • Container Port: 3000 • KS &amp; RBF-MMD Monitoring Engine
          </span>
        </div>
      </footer>
    </div>
  );
}
