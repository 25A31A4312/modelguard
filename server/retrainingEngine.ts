import type {
  HumanReviewSample,
  RetrainingValidationReport,
} from '../src/types/modelguard.ts';
import type { TrainedClassifierCheckpoint } from './imageAnalysis.ts';

export interface LabeledFeatureSample {
  id: string;
  label: string;
  features: number[];
}

export interface RetrainingExecutionResult {
  success: boolean;
  failureReason?:
    | 'INSUFFICIENT_LABELED_DATA'
    | 'INSUFFICIENT_CLASS_DIVERSITY'
    | 'INSUFFICIENT_VALIDATION_SPLIT';
  errorMessage?: string;
  eligibleSamplesCount: number;
  trainSamplesCount: number;
  valSamplesCount: number;
  minRequiredSamples: number;
  epochsCompleted: number;
  totalEpochs: number;
  finalTrainLoss: number | null;
  finalValLoss: number | null;
  lossHistory: { epoch: number; trainLoss: number; valLoss: number }[];
  durationMs: number;
  checkpoint?: TrainedClassifierCheckpoint;
  validationReport?: RetrainingValidationReport;
}

/**
 * Extracts only genuine, human-reviewed samples with confirmed ground-truth labels
 * (`status === 'ACCEPTED' || status === 'OVERRIDDEN'`), `sendToRetraining === true`,
 * and valid finite feature vectors. Rejected, pending, or unlabeled items are strictly excluded.
 */
export function collectEligibleRetrainingSamples(
  reviewQueue: HumanReviewSample[]
): LabeledFeatureSample[] {
  const eligible: LabeledFeatureSample[] = [];

  for (const item of reviewQueue) {
    if (!item.reviewed) continue;
    if (!item.sendToRetraining) continue;
    if (item.status !== 'ACCEPTED' && item.status !== 'OVERRIDDEN') continue;
    const label = item.groundTruthLabel?.trim().toLowerCase();
    if (!label) continue;
    if (
      !Array.isArray(item.featureVector) ||
      item.featureVector.length === 0 ||
      !item.featureVector.every((v) => typeof v === 'number' && Number.isFinite(v))
    ) {
      continue;
    }

    eligible.push({
      id: item.id,
      label,
      features: [...item.featureVector],
    });
  }

  return eligible;
}

function softmax(logits: number[]): number[] {
  const maxL = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - maxL));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / sum);
}

function evaluateSplitMetrics(
  samples: LabeledFeatureSample[],
  classes: string[],
  featureMeans: number[],
  featureStds: number[],
  weights: number[][],
  biases: number[],
  l2Lambda: number
): {
  loss: number;
  accuracy: number;
  macroPrecision: number;
  macroRecall: number;
  macroF1: number;
} {
  const K = classes.length;
  const classIndex = new Map<string, number>();
  classes.forEach((c, idx) => classIndex.set(c, idx));

  let totalCrossEntropy = 0;
  let correct = 0;

  const tp = new Array(K).fill(0);
  const fp = new Array(K).fill(0);
  const fn = new Array(K).fill(0);

  for (const sample of samples) {
    const yTrue = classIndex.get(sample.label) ?? 0;
    const z = sample.features.map((v, d) => (v - featureMeans[d]) / featureStds[d]);

    const logits = new Array(K).fill(0);
    for (let k = 0; k < K; k++) {
      let s = biases[k];
      for (let d = 0; d < z.length; d++) {
        s += weights[k][d] * z[d];
      }
      logits[k] = s;
    }

    const probs = softmax(logits);
    const pTrue = Math.max(1e-12, probs[yTrue]);
    totalCrossEntropy += -Math.log(pTrue);

    let yPred = 0;
    let bestP = -1;
    for (let k = 0; k < K; k++) {
      if (probs[k] > bestP) {
        bestP = probs[k];
        yPred = k;
      }
    }

    if (yPred === yTrue) {
      correct++;
      tp[yTrue]++;
    } else {
      fp[yPred]++;
      fn[yTrue]++;
    }
  }

  let wSqSum = 0;
  for (let k = 0; k < K; k++) {
    for (let d = 0; d < weights[k].length; d++) {
      wSqSum += weights[k][d] * weights[k][d];
    }
  }

  const loss = Number((totalCrossEntropy / Math.max(1, samples.length) + 0.5 * l2Lambda * wSqSum).toFixed(4));
  const accuracy = Number((correct / Math.max(1, samples.length)).toFixed(4));

  let sumP = 0;
  let sumR = 0;
  let sumF1 = 0;
  let activeClassCount = 0;

  for (let k = 0; k < K; k++) {
    const hasSupport = tp[k] + fn[k] > 0 || fp[k] > 0;
    if (!hasSupport) continue;
    activeClassCount++;
    const p = tp[k] + fp[k] > 0 ? tp[k] / (tp[k] + fp[k]) : 0;
    const r = tp[k] + fn[k] > 0 ? tp[k] / (tp[k] + fn[k]) : 0;
    const f1 = p + r > 0 ? (2 * p * r) / (p + r) : 0;
    sumP += p;
    sumR += r;
    sumF1 += f1;
  }

  const denom = Math.max(1, activeClassCount);
  return {
    loss,
    accuracy,
    macroPrecision: Number((sumP / denom).toFixed(4)),
    macroRecall: Number((sumR / denom).toFixed(4)),
    macroF1: Number((sumF1 / denom).toFixed(4)),
  };
}

/**
 * Trains a genuine L2-regularized Multinomial Softmax Logistic Regression classifier
 * on extracted image feature vectors from human-reviewed samples, using a deterministic
 * stratified train/validation holdout split and evaluating against the prior checkpoint.
 */
export function runSupervisedFeatureRetraining(options: {
  eligibleSamples: LabeledFeatureSample[];
  candidateVersion: string;
  minRequiredSamples?: number;
  totalEpochs?: number;
  learningRate?: number;
  l2Lambda?: number;
  previousCheckpoint?: TrainedClassifierCheckpoint | null;
}): RetrainingExecutionResult {
  const startTime = performance.now();
  const minRequiredSamples =
    options.minRequiredSamples ?? Math.max(4, Number(process.env.RETRAIN_MIN_LABELED_SAMPLES) || 6);
  const totalEpochs = options.totalEpochs ?? 30;
  const learningRate = options.learningRate ?? 0.2;
  const l2Lambda = options.l2Lambda ?? 0.01;
  const samples = options.eligibleSamples;

  if (samples.length < minRequiredSamples) {
    return {
      success: false,
      failureReason: 'INSUFFICIENT_LABELED_DATA',
      errorMessage: `Retraining aborted: Only ${samples.length} eligible human-reviewed labeled sample(s) available; minimum ${minRequiredSamples} required for train/validation split.`,
      eligibleSamplesCount: samples.length,
      trainSamplesCount: 0,
      valSamplesCount: 0,
      minRequiredSamples,
      epochsCompleted: 0,
      totalEpochs,
      finalTrainLoss: null,
      finalValLoss: null,
      lossHistory: [],
      durationMs: Math.max(1, Math.round(performance.now() - startTime)),
    };
  }

  // Group by class for deterministic stratified train/validation split
  const byClass = new Map<string, LabeledFeatureSample[]>();
  for (const s of samples) {
    const arr = byClass.get(s.label) || [];
    arr.push(s);
    byClass.set(s.label, arr);
  }

  const classes = Array.from(byClass.keys()).sort();
  if (classes.length < 2) {
    return {
      success: false,
      failureReason: 'INSUFFICIENT_CLASS_DIVERSITY',
      errorMessage: `Retraining aborted: Eligible reviewed samples contain only ${classes.length} class (${classes[0] || 'none'}). At least 2 distinct ground-truth classes are required to train a discriminative classifier.`,
      eligibleSamplesCount: samples.length,
      trainSamplesCount: 0,
      valSamplesCount: 0,
      minRequiredSamples,
      epochsCompleted: 0,
      totalEpochs,
      finalTrainLoss: null,
      finalValLoss: null,
      lossHistory: [],
      durationMs: Math.max(1, Math.round(performance.now() - startTime)),
    };
  }

  const trainSet: LabeledFeatureSample[] = [];
  const valSet: LabeledFeatureSample[] = [];

  for (const cls of classes) {
    const group = byClass.get(cls)!;
    if (group.length >= 2) {
      const valCount = Math.max(1, Math.round(group.length * 0.3));
      const actualVal = Math.min(group.length - 1, valCount);
      valSet.push(...group.slice(0, actualVal));
      trainSet.push(...group.slice(actualVal));
    } else {
      // Single-instance class goes to trainSet
      trainSet.push(group[0]);
    }
  }

  // Ensure at least 2 validation samples
  if (valSet.length < 2 && trainSet.length > 2) {
    valSet.push(trainSet.pop()!);
  }

  if (trainSet.length < 2 || valSet.length < 2) {
    return {
      success: false,
      failureReason: 'INSUFFICIENT_VALIDATION_SPLIT',
      errorMessage: `Retraining aborted: Unable to construct non-trivial train (${trainSet.length}) and held-out validation (${valSet.length}) splits.`,
      eligibleSamplesCount: samples.length,
      trainSamplesCount: trainSet.length,
      valSamplesCount: valSet.length,
      minRequiredSamples,
      epochsCompleted: 0,
      totalEpochs,
      finalTrainLoss: null,
      finalValLoss: null,
      lossHistory: [],
      durationMs: Math.max(1, Math.round(performance.now() - startTime)),
    };
  }

  const dim = trainSet[0].features.length;
  const K = classes.length;

  // Compute feature standardization strictly on trainSet (zero validation leakage)
  const featureMeans = new Array(dim).fill(0);
  const featureStds = new Array(dim).fill(1);

  for (let d = 0; d < dim; d++) {
    let sum = 0;
    for (const s of trainSet) sum += s.features[d];
    const mu = sum / trainSet.length;
    featureMeans[d] = Number(mu.toFixed(6));

    let varSum = 0;
    for (const s of trainSet) {
      const diff = s.features[d] - mu;
      varSum += diff * diff;
    }
    const sd = trainSet.length > 1 ? Math.sqrt(varSum / (trainSet.length - 1)) : 0;
    featureStds[d] = sd > 1e-8 ? Number(sd.toFixed(6)) : 1.0;
  }

  // Evaluate baseline on held-out valSet:
  // If previousCheckpoint shares compatible classes/dimensions, evaluate its mapped weights; otherwise zero-weight uniform prior baseline
  const zeroWeights = Array.from({ length: K }, () => new Array(dim).fill(0));
  const zeroBiases = new Array(K).fill(0);

  let baselineEval = evaluateSplitMetrics(
    valSet,
    classes,
    featureMeans,
    featureStds,
    zeroWeights,
    zeroBiases,
    l2Lambda
  );

  if (
    options.previousCheckpoint &&
    options.previousCheckpoint.featureMeans.length === dim
  ) {
    // Evaluate previous checkpoint directly on valSet
    let prevCorrect = 0;
    let prevLossSum = 0;
    const tp = new Array(K).fill(0);
    const fp = new Array(K).fill(0);
    const fn = new Array(K).fill(0);

    for (const s of valSet) {
      const trueIdx = classes.indexOf(s.label);
      const zPrev = s.features.map(
        (v, d) =>
          (v - options.previousCheckpoint!.featureMeans[d]) /
          (options.previousCheckpoint!.featureStds[d] || 1)
      );
      const logitsPrev = options.previousCheckpoint.classes.map((_, pk) => {
        let dot = options.previousCheckpoint!.biases[pk] || 0;
        for (let d = 0; d < dim; d++) {
          dot += (options.previousCheckpoint!.weights[pk]?.[d] || 0) * zPrev[d];
        }
        return dot;
      });
      const probsPrev = softmax(logitsPrev);
      const prevClassIdx = options.previousCheckpoint.classes.indexOf(s.label);
      const pTrue = prevClassIdx >= 0 ? Math.max(1e-12, probsPrev[prevClassIdx]) : 1 / Math.max(2, K);
      prevLossSum += -Math.log(pTrue);

      let bestPk = 0;
      for (let pk = 1; pk < probsPrev.length; pk++) {
        if (probsPrev[pk] > probsPrev[bestPk]) bestPk = pk;
      }
      const predLabel = options.previousCheckpoint.classes[bestPk];
      const predIdxInCurr = classes.indexOf(predLabel);

      if (predLabel === s.label) {
        prevCorrect++;
        if (trueIdx >= 0) tp[trueIdx]++;
      } else {
        if (predIdxInCurr >= 0) fp[predIdxInCurr]++;
        if (trueIdx >= 0) fn[trueIdx]++;
      }
    }

    let sumP = 0;
    let sumR = 0;
    let sumF1 = 0;
    for (let k = 0; k < K; k++) {
      const p = tp[k] + fp[k] > 0 ? tp[k] / (tp[k] + fp[k]) : 0;
      const r = tp[k] + fn[k] > 0 ? tp[k] / (tp[k] + fn[k]) : 0;
      const f1 = p + r > 0 ? (2 * p * r) / (p + r) : 0;
      sumP += p;
      sumR += r;
      sumF1 += f1;
    }

    baselineEval = {
      loss: Number((prevLossSum / valSet.length).toFixed(4)),
      accuracy: Number((prevCorrect / valSet.length).toFixed(4)),
      macroPrecision: Number((sumP / K).toFixed(4)),
      macroRecall: Number((sumR / K).toFixed(4)),
      macroF1: Number((sumF1 / K).toFixed(4)),
    };
  }

  // Initialize weights deterministically to 0 and optimize via batch gradient descent on standardized trainSet
  const weights = Array.from({ length: K }, () => new Array(dim).fill(0));
  const biases = new Array(K).fill(0);
  const classToIdx = new Map<string, number>();
  classes.forEach((c, idx) => classToIdx.set(c, idx));

  const standardizedTrain = trainSet.map((s) => ({
    y: classToIdx.get(s.label)!,
    z: s.features.map((v, d) => (v - featureMeans[d]) / featureStds[d]),
  }));

  const lossHistory: { epoch: number; trainLoss: number; valLoss: number }[] = [];

  for (let epoch = 1; epoch <= totalEpochs; epoch++) {
    const gradW = Array.from({ length: K }, () => new Array(dim).fill(0));
    const gradB = new Array(K).fill(0);

    for (const item of standardizedTrain) {
      const logits = new Array(K).fill(0);
      for (let k = 0; k < K; k++) {
        let dot = biases[k];
        for (let d = 0; d < dim; d++) dot += weights[k][d] * item.z[d];
        logits[k] = dot;
      }
      const probs = softmax(logits);
      for (let k = 0; k < K; k++) {
        const err = probs[k] - (k === item.y ? 1.0 : 0.0);
        gradB[k] += err;
        for (let d = 0; d < dim; d++) {
          gradW[k][d] += err * item.z[d];
        }
      }
    }

    const nTrain = standardizedTrain.length;
    for (let k = 0; k < K; k++) {
      biases[k] -= learningRate * (gradB[k] / nTrain);
      for (let d = 0; d < dim; d++) {
        const regGrad = gradW[k][d] / nTrain + l2Lambda * weights[k][d];
        weights[k][d] -= learningRate * regGrad;
      }
    }

    const trainEval = evaluateSplitMetrics(
      trainSet,
      classes,
      featureMeans,
      featureStds,
      weights,
      biases,
      l2Lambda
    );
    const valEvalStep = evaluateSplitMetrics(
      valSet,
      classes,
      featureMeans,
      featureStds,
      weights,
      biases,
      l2Lambda
    );

    lossHistory.push({
      epoch,
      trainLoss: trainEval.loss,
      valLoss: valEvalStep.loss,
    });
  }

  const finalTrainEval = evaluateSplitMetrics(
    trainSet,
    classes,
    featureMeans,
    featureStds,
    weights,
    biases,
    l2Lambda
  );
  const candidateEval = evaluateSplitMetrics(
    valSet,
    classes,
    featureMeans,
    featureStds,
    weights,
    biases,
    l2Lambda
  );

  // Strict Validation Gate:
  // Candidate must achieve finite validation loss, >= 50% held-out accuracy, and not regress below baseline accuracy or macro-F1.
  const nonRegressing =
    candidateEval.accuracy >= baselineEval.accuracy &&
    candidateEval.macroF1 >= baselineEval.macroF1;
  const minimumQuality =
    Number.isFinite(candidateEval.loss) && candidateEval.accuracy >= 0.5;
  const passedPromotionCriteria = nonRegressing && minimumQuality;

  const promotionPolicyReason = passedPromotionCriteria
    ? `Validation gate passed on ${valSet.length} held-out sample(s): Accuracy ${(candidateEval.accuracy * 100).toFixed(1)}% (baseline ${(baselineEval.accuracy * 100).toFixed(1)}%), Macro-F1 ${(candidateEval.macroF1 * 100).toFixed(1)}% (baseline ${(baselineEval.macroF1 * 100).toFixed(1)}%), Val Loss ${candidateEval.loss}.`
    : `Validation gate failed on ${valSet.length} held-out sample(s): Accuracy ${(candidateEval.accuracy * 100).toFixed(1)}% vs baseline ${(baselineEval.accuracy * 100).toFixed(1)}% (min 50.0% & non-regression required). Candidate blocked from promotion.`;

  const checkpoint: TrainedClassifierCheckpoint = {
    version: options.candidateVersion,
    classes,
    featureMeans,
    featureStds,
    weights: weights.map((row) => row.map((w) => Number(w.toFixed(6)))),
    biases: biases.map((b) => Number(b.toFixed(6))),
  };

  const validationReport: RetrainingValidationReport = {
    baselineAccuracy: baselineEval.accuracy,
    candidateAccuracy: candidateEval.accuracy,
    baselineMacroPrecision: baselineEval.macroPrecision,
    candidateMacroPrecision: candidateEval.macroPrecision,
    baselineMacroRecall: baselineEval.macroRecall,
    candidateMacroRecall: candidateEval.macroRecall,
    baselineMacroF1: baselineEval.macroF1,
    candidateMacroF1: candidateEval.macroF1,
    baselineValLoss: baselineEval.loss,
    candidateValLoss: candidateEval.loss,
    valSampleCount: valSet.length,
    distinctClasses: classes,
    passedPromotionCriteria,
    promotionPolicyReason,
    baselineMAP: baselineEval.macroF1,
    candidateMAP: candidateEval.macroF1,
  };

  return {
    success: true,
    eligibleSamplesCount: samples.length,
    trainSamplesCount: trainSet.length,
    valSamplesCount: valSet.length,
    minRequiredSamples,
    epochsCompleted: totalEpochs,
    totalEpochs,
    finalTrainLoss: finalTrainEval.loss,
    finalValLoss: candidateEval.loss,
    lossHistory,
    durationMs: Math.max(1, Math.round(performance.now() - startTime)),
    checkpoint,
    validationReport,
  };
}
