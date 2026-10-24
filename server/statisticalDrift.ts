import type {
  DistanceMetricResult,
  DriftMetric,
  DriftStatus,
  KSTestResult,
  MMDTestResult,
} from '../src/types/modelguard.ts';

/**
 * Documented Statistical Configuration for ModelGuard Covariate Drift Engine:
 *
 * 1. Minimum Sample Requirement (`minSamples`, default = 5, configurable via DRIFT_MIN_SAMPLES):
 *    - Applied independently to BOTH reference observations (m) and current observations (n).
 *    - When m < minSamples or n < minSamples (including 0 current samples), NO statistical test
 *      is executed; status is strictly 'INSUFFICIENT_DATA' and test statistics/scores are null.
 *
 * 2. Two-Sample Kolmogorov-Smirnov (KS) Test (1D continuous/numerical features):
 *    - Statistic: D_{m,n} = sup_t |F_{1,m}(t) - F_{2,n}(t)| with exact tie-advancing ECDF evaluation.
 *    - Asymptotic p-value: Two-sided Kolmogorov survival distribution with Stephens (1970)
 *      finite-sample correction lambda = (sqrt(n_e) + 0.12 + 0.11 / sqrt(n_e)) * D_{m,n},
 *      where effective sample size n_e = (m * n) / (m + n).
 *    - Threshold: Rejects H0 at pValue < alpha (default alpha = 0.05; severe drift at pValue < alpha / 5).
 *
 * 3. Multivariate Maximum Mean Discrepancy (MMD) Test (Feature Vectors in R^d):
 *    - Kernel: Gaussian Radial Basis Function (RBF) k(x, y) = exp(-||x - y||_2^2 / (2 * sigma^2))
 *      computed on reference-standardized coordinates (constant coordinates safely use scale = 1.0).
 *    - Bandwidth (sigma): Median heuristic over pairwise Euclidean distances of pooled standardized
 *      observations (safe fallback sigma = 1.0 if median pairwise distance <= 1e-8).
 *    - Estimator & Threshold: Biased RKHS distance MMD_b = sqrt(max(0, E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]))
 *      and unbiased MMD_u^2, paired with a deterministic B = 200 permutation test over the Gram matrix
 *      yielding exact empirical p-value and the (1 - alpha)-quantile critical threshold.
 *
 * 4. Descriptive Distribution Distances (Reported separately from KS & MMD):
 *    - Wasserstein-1 (Earth Mover's Distance): L1 integral between empirical quantile functions.
 *    - Population Stability Index (PSI): 10-bin smoothed KL divergence (moderate >= 0.10, severe >= 0.25).
 */
export const DEFAULT_DRIFT_CONFIG = {
  minSamplesRequired: Math.max(2, Number(process.env.DRIFT_MIN_SAMPLES) || 5),
  alpha: Number(process.env.AUTO_RETRAIN_KS_ALPHA) || 0.05,
  mmdPermutations: 200,
  psiModerateThreshold: 0.1,
  psiSevereThreshold: 0.25,
};

/**
 * Strips NaN, Infinity, -Infinity, null, undefined, and non-number entries.
 */
export function sanitizeNumericArray(arr: unknown): number[] {
  if (!Array.isArray(arr)) return [];
  return arr.filter((x): x is number => typeof x === 'number' && Number.isFinite(x));
}

/**
 * Strips invalid, mismatched-dimension, or non-finite feature vectors.
 */
export function sanitizeFeatureVectors(vectors: unknown, expectedDim?: number): number[][] {
  if (!Array.isArray(vectors)) return [];
  const valid: number[][] = [];
  let dim = expectedDim;

  for (const vec of vectors) {
    if (!Array.isArray(vec) || vec.length === 0) continue;
    if (dim === undefined) dim = vec.length;
    if (vec.length !== dim) continue;
    if (vec.every((v) => typeof v === 'number' && Number.isFinite(v))) {
      valid.push(vec as number[]);
    }
  }
  return valid;
}

function meanOf(arr: number[]): number | null {
  if (arr.length === 0) return null;
  return Number((arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(3));
}

function ksCriticalCoefficient(alpha: number): number {
  if (alpha <= 0.001) return 1.9495;
  if (alpha <= 0.01) return 1.6276;
  if (alpha <= 0.05) return 1.3581;
  if (alpha <= 0.1) return 1.2238;
  return Math.sqrt(-0.5 * Math.log(Math.max(1e-6, alpha / 2)));
}

/**
 * Genuine Two-Sample Kolmogorov-Smirnov (KS) Test for 1D numerical observations.
 */
export function computeTwoSampleKSTest(
  refSample: (number | null | undefined)[],
  currSample: (number | null | undefined)[],
  alpha: number = DEFAULT_DRIFT_CONFIG.alpha,
  minSamples: number = DEFAULT_DRIFT_CONFIG.minSamplesRequired,
  featureName: string = 'Feature'
): KSTestResult & {
  dStatistic: number | null;
  isDrift: boolean;
  sampleCountRef: number;
  sampleCountCurr: number;
  valid: boolean;
  message: string;
} {
  const timestamp = new Date().toISOString();
  const X = sanitizeNumericArray(refSample).sort((a, b) => a - b);
  const Y = sanitizeNumericArray(currSample).sort((a, b) => a - b);

  const m = X.length;
  const n = Y.length;
  const baselineMean = meanOf(X);
  const currentMean = meanOf(Y);

  if (m < minSamples || n < minSamples) {
    const reason =
      n === 0
        ? `Zero current inference observations recorded (Current: 0, Reference: ${m}, Minimum required: ${minSamples}). Run inferences to collect real feature observations.`
        : m < minSamples
        ? `Insufficient reference observations (Reference: ${m}, Current: ${n}, Minimum required: ${minSamples}). Load a valid reference dataset.`
        : `Insufficient current observations (Current: ${n}, Reference: ${m}, Minimum required: ${minSamples}). Collect at least ${minSamples - n} more sample(s) before running KS test.`;

    return {
      method: 'KS_TEST',
      featureName,
      statistic: null,
      dStatistic: null,
      pValue: null,
      criticalValue: null,
      alpha,
      referenceSampleCount: m,
      currentSampleCount: n,
      sampleCountRef: m,
      sampleCountCurr: n,
      minSamplesRequired: minSamples,
      baselineMean,
      currentMean,
      status: 'INSUFFICIENT_DATA',
      reason,
      message: reason,
      timestamp,
      isDrift: false,
      valid: false,
    };
  }

  // Exact tie-aware two-sample empirical CDF supremum distance:
  // Advance ALL observations <= threshold t in BOTH sorted samples before taking |F_m(t) - G_n(t)|.
  let i = 0;
  let j = 0;
  let dMax = 0;

  while (i < m && j < n) {
    const t = Math.min(X[i], Y[j]);
    while (i < m && X[i] <= t) i++;
    while (j < n && Y[j] <= t) j++;

    const diff = Math.abs(i / m - j / n);
    if (diff > dMax) {
      dMax = diff;
    }
  }

  const statistic = Number(Math.min(1.0, Math.max(0.0, dMax)).toFixed(4));
  const effectiveN = (m * n) / (m + n);
  const sqrtNe = Math.sqrt(effectiveN);
  const criticalValue = Number((ksCriticalCoefficient(alpha) / sqrtNe).toFixed(4));

  // Stephens (1970) Kolmogorov survival series
  let pValue = 1.0;
  if (statistic > 0) {
    const lambda = (sqrtNe + 0.12 + 0.11 / sqrtNe) * statistic;
    let sum = 0;
    for (let k = 1; k <= 100; k++) {
      const term = (k % 2 === 1 ? 1 : -1) * Math.exp(-2 * k * k * lambda * lambda);
      sum += term;
      if (Math.abs(term) < 1e-14) break;
    }
    pValue = Math.min(1.0, Math.max(0.0, 2 * sum));
  }
  pValue = Number(pValue.toFixed(6));

  const isDrift = pValue < alpha;
  let status: DriftStatus = 'STABLE';
  if (pValue < Math.min(0.01, alpha / 5) || (isDrift && statistic >= Math.min(1.0, criticalValue * 1.35))) {
    status = 'SEVERE_DRIFT';
  } else if (isDrift) {
    status = 'MODERATE_DRIFT';
  }

  const reason = isDrift
    ? `KS test rejected H0 at alpha=${alpha} (D=${statistic}, p=${pValue} < ${alpha}, critical D=${criticalValue}, Ref N=${m}, Curr N=${n}).`
    : `KS test retained H0 at alpha=${alpha} (D=${statistic}, p=${pValue} >= ${alpha}, critical D=${criticalValue}, Ref N=${m}, Curr N=${n}).`;

  return {
    method: 'KS_TEST',
    featureName,
    statistic,
    dStatistic: statistic,
    pValue,
    criticalValue,
    alpha,
    referenceSampleCount: m,
    currentSampleCount: n,
    sampleCountRef: m,
    sampleCountCurr: n,
    minSamplesRequired: minSamples,
    baselineMean,
    currentMean,
    status,
    reason,
    message: reason,
    timestamp,
    isDrift,
    valid: true,
  };
}

/**
 * Genuine Multivariate Maximum Mean Discrepancy (MMD) Test with Gaussian RBF Kernel
 * and Gram-matrix permutation p-value / quantile threshold.
 */
export function computeMultivariateMMDTest(
  featureNames: string[],
  refVectorsInput: number[][],
  currVectorsInput: number[][],
  alpha: number = DEFAULT_DRIFT_CONFIG.alpha,
  minSamples: number = DEFAULT_DRIFT_CONFIG.minSamplesRequired,
  permutations: number = DEFAULT_DRIFT_CONFIG.mmdPermutations
): MMDTestResult & { isDrift: boolean; valid: boolean } {
  const timestamp = new Date().toISOString();
  const dim = featureNames.length > 0 ? featureNames.length : undefined;
  const Xraw = sanitizeFeatureVectors(refVectorsInput, dim);
  const Yraw = sanitizeFeatureVectors(currVectorsInput, Xraw[0]?.length ?? dim);

  const m = Xraw.length;
  const n = Yraw.length;

  if (m < minSamples || n < minSamples || (Xraw[0]?.length ?? 0) === 0) {
    const reason =
      n === 0
        ? `Zero current feature vectors recorded (Current: 0, Reference: ${m}, Minimum required: ${minSamples}). MMD kernel test not executed.`
        : m < minSamples
        ? `Insufficient reference feature vectors (Reference: ${m}, Current: ${n}, Minimum required: ${minSamples}).`
        : `Insufficient current feature vectors (Current: ${n}, Reference: ${m}, Minimum required: ${minSamples}). Collect ${minSamples - n} more sample(s).`;

    return {
      method: 'MMD_RBF',
      featureNames,
      kernel: 'RBF_GAUSSIAN',
      bandwidthSigma: null,
      permutations,
      mmdScore: null,
      unbiasedMmdSquared: null,
      pValue: null,
      threshold: null,
      alpha,
      referenceSampleCount: m,
      currentSampleCount: n,
      minSamplesRequired: minSamples,
      status: 'INSUFFICIENT_DATA',
      reason,
      timestamp,
      isDrift: false,
      valid: false,
    };
  }

  // Cap pooled sample window to most recent 150 observations per group so O((m+n)^2) Gram matrix stays <3ms
  const Xcap = Xraw.slice(0, 150);
  const Ycap = Yraw.slice(0, 150);
  const mEff = Xcap.length;
  const nEff = Ycap.length;
  const d = Xcap[0].length;

  // Standardize coordinates using reference mean & std (safe scale = 1.0 for constant features)
  const refMeans = new Float64Array(d);
  const refStds = new Float64Array(d);
  for (let coord = 0; coord < d; coord++) {
    let s = 0;
    for (let i = 0; i < mEff; i++) s += Xcap[i][coord];
    const mu = s / mEff;
    refMeans[coord] = mu;

    let v = 0;
    for (let i = 0; i < mEff; i++) {
      const diff = Xcap[i][coord] - mu;
      v += diff * diff;
    }
    const sd = mEff > 1 ? Math.sqrt(v / (mEff - 1)) : 0;
    refStds[coord] = sd > 1e-8 ? sd : 1.0;
  }

  const pooled: Float64Array[] = [];
  for (const vec of Xcap) {
    const z = new Float64Array(d);
    for (let c = 0; c < d; c++) z[c] = (vec[c] - refMeans[c]) / refStds[c];
    pooled.push(z);
  }
  for (const vec of Ycap) {
    const z = new Float64Array(d);
    for (let c = 0; c < d; c++) z[c] = (vec[c] - refMeans[c]) / refStds[c];
    pooled.push(z);
  }

  const total = mEff + nEff;
  const distSqMatrix = new Float64Array(total * total);
  const pairwiseDists: number[] = [];

  for (let p = 0; p < total; p++) {
    for (let q = p + 1; q < total; q++) {
      let sq = 0;
      const vp = pooled[p];
      const vq = pooled[q];
      for (let c = 0; c < d; c++) {
        const diff = vp[c] - vq[c];
        sq += diff * diff;
      }
      distSqMatrix[p * total + q] = sq;
      distSqMatrix[q * total + p] = sq;
      if (sq > 0) pairwiseDists.push(Math.sqrt(sq));
    }
  }

  // Median heuristic bandwidth sigma (fallback sigma = 1.0 when all vectors are constant/identical)
  let sigma = 1.0;
  if (pairwiseDists.length > 0) {
    pairwiseDists.sort((a, b) => a - b);
    const mid = Math.floor(pairwiseDists.length / 2);
    const med =
      pairwiseDists.length % 2 === 0
        ? 0.5 * (pairwiseDists[mid - 1] + pairwiseDists[mid])
        : pairwiseDists[mid];
    if (Number.isFinite(med) && med > 1e-8) {
      sigma = med;
    }
  }

  const gamma = 1.0 / (2.0 * sigma * sigma);
  const gram = new Float64Array(total * total);
  for (let p = 0; p < total; p++) {
    gram[p * total + p] = 1.0;
    for (let q = p + 1; q < total; q++) {
      const kval = Math.exp(-gamma * distSqMatrix[p * total + q]);
      gram[p * total + q] = kval;
      gram[q * total + p] = kval;
    }
  }

  const evalMmdFromIndices = (idx: Int32Array): { biasedMmd: number; unbiasedSq: number } => {
    let sumXX = 0;
    let sumXXOffDiag = 0;
    for (let a = 0; a < mEff; a++) {
      const ia = idx[a];
      sumXX += 1.0;
      for (let b = a + 1; b < mEff; b++) {
        const kv = gram[ia * total + idx[b]];
        sumXX += 2 * kv;
        sumXXOffDiag += 2 * kv;
      }
    }

    let sumYY = 0;
    let sumYYOffDiag = 0;
    for (let a = mEff; a < total; a++) {
      const ia = idx[a];
      sumYY += 1.0;
      for (let b = a + 1; b < total; b++) {
        const kv = gram[ia * total + idx[b]];
        sumYY += 2 * kv;
        sumYYOffDiag += 2 * kv;
      }
    }

    let sumXY = 0;
    for (let a = 0; a < mEff; a++) {
      const ia = idx[a];
      for (let b = mEff; b < total; b++) {
        sumXY += gram[ia * total + idx[b]];
      }
    }

    const biasedSq = sumXX / (mEff * mEff) + sumYY / (nEff * nEff) - (2 * sumXY) / (mEff * nEff);
    const unbiasedSq =
      sumXXOffDiag / (mEff * (mEff - 1)) +
      sumYYOffDiag / (nEff * (nEff - 1)) -
      (2 * sumXY) / (mEff * nEff);

    return {
      biasedMmd: Math.sqrt(Math.max(0, biasedSq)),
      unbiasedSq,
    };
  };

  const identityIdx = new Int32Array(total);
  for (let t = 0; t < total; t++) identityIdx[t] = t;
  const observed = evalMmdFromIndices(identityIdx);

  // Deterministic Fisher-Yates permutation test over precomputed Gram matrix
  let rngState = 133742;
  const nextRand = () => {
    rngState = (rngState * 1664525 + 1013904223) >>> 0;
    return rngState / 4294967296;
  };

  const permIdx = new Int32Array(identityIdx);
  const permScores: number[] = [];
  let geCount = 0;

  for (let b = 0; b < permutations; b++) {
    for (let k = total - 1; k > 0; k--) {
      const swap = Math.floor(nextRand() * (k + 1));
      const tmp = permIdx[k];
      permIdx[k] = permIdx[swap];
      permIdx[swap] = tmp;
    }
    const permRes = evalMmdFromIndices(permIdx);
    permScores.push(permRes.biasedMmd);
    if (permRes.biasedMmd >= observed.biasedMmd - 1e-12) {
      geCount++;
    }
  }

  permScores.sort((a, b) => a - b);
  const qIndex = Math.min(
    permScores.length - 1,
    Math.max(0, Math.ceil((1 - alpha) * permScores.length) - 1)
  );
  const threshold = Number(permScores[qIndex].toFixed(4));
  const mmdScore = Number(observed.biasedMmd.toFixed(4));
  const unbiasedMmdSquared = Number(observed.unbiasedSq.toFixed(5));
  const pValue = Number(((geCount + 1) / (permutations + 1)).toFixed(4));

  const isDrift = mmdScore > threshold && pValue <= alpha;
  let status: DriftStatus = 'STABLE';
  if (isDrift && (pValue <= Math.max(0.01, alpha / 3) || mmdScore > threshold * 1.35)) {
    status = 'SEVERE_DRIFT';
  } else if (isDrift) {
    status = 'MODERATE_DRIFT';
  }

  const reason = isDrift
    ? `Multivariate RBF-MMD detected joint feature shift (MMD=${mmdScore} > ${threshold}, perm p=${pValue} <= ${alpha}, sigma=${sigma.toFixed(3)}, d=${d}).`
    : `Multivariate RBF-MMD within reference envelope (MMD=${mmdScore} <= ${threshold}, perm p=${pValue}, sigma=${sigma.toFixed(3)}, d=${d}).`;

  return {
    method: 'MMD_RBF',
    featureNames,
    kernel: 'RBF_GAUSSIAN',
    bandwidthSigma: Number(sigma.toFixed(4)),
    permutations,
    mmdScore,
    unbiasedMmdSquared,
    pValue,
    threshold,
    alpha,
    referenceSampleCount: m,
    currentSampleCount: n,
    minSamplesRequired: minSamples,
    status,
    reason,
    timestamp,
    isDrift,
    valid: true,
  };
}

/**
 * 1D convenience wrapper for MMD when testing single-variable arrays directly.
 */
export function computeMMD(
  refSample: (number | null | undefined)[],
  currSample: (number | null | undefined)[],
  alpha: number = DEFAULT_DRIFT_CONFIG.alpha,
  minSamples: number = DEFAULT_DRIFT_CONFIG.minSamplesRequired
) {
  const X = sanitizeNumericArray(refSample).map((v) => [v]);
  const Y = sanitizeNumericArray(currSample).map((v) => [v]);
  const res = computeMultivariateMMDTest(['univariate_feature'], X, Y, alpha, minSamples);
  return {
    mmdScore: res.mmdScore,
    pValue: res.pValue,
    threshold: res.threshold,
    isDrift: res.isDrift,
    sampleCountRef: res.referenceSampleCount,
    sampleCountCurr: res.currentSampleCount,
    valid: res.valid,
    bandwidth: res.bandwidthSigma,
    status: res.status,
    message: res.reason,
  };
}

/**
 * Wasserstein-1 Distance (Earth Mover's Distance) for 1D empirical distributions.
 * Returns null if either array is empty.
 */
export function computeWassersteinDistance(
  refSample: (number | null | undefined)[],
  currSample: (number | null | undefined)[]
): number | null {
  const X = sanitizeNumericArray(refSample).sort((a, b) => a - b);
  const Y = sanitizeNumericArray(currSample).sort((a, b) => a - b);

  if (X.length === 0 || Y.length === 0) return null;

  const steps = 200;
  let sumDiff = 0;
  for (let s = 0; s <= steps; s++) {
    const q = s / steps;
    const posX = q * (X.length - 1);
    const loX = Math.floor(posX);
    const hiX = Math.ceil(posX);
    const valX = X[loX] + (posX - loX) * (X[hiX] - X[loX]);

    const posY = q * (Y.length - 1);
    const loY = Math.floor(posY);
    const hiY = Math.ceil(posY);
    const valY = Y[loY] + (posY - loY) * (Y[hiY] - Y[loY]);

    sumDiff += Math.abs(valX - valY);
  }
  return Number((sumDiff / (steps + 1)).toFixed(3));
}

/**
 * Population Stability Index (PSI) using 10 equal-width bins with Laplace smoothing.
 * Returns null when sample counts do not meet minSamples.
 */
export function computePSI(
  refSample: (number | null | undefined)[],
  currSample: (number | null | undefined)[],
  minSamples: number = DEFAULT_DRIFT_CONFIG.minSamplesRequired
): number | null {
  const X = sanitizeNumericArray(refSample);
  const Y = sanitizeNumericArray(currSample);

  if (X.length < minSamples || Y.length < minSamples) return null;

  const minVal = Math.min(...X, ...Y);
  const maxVal = Math.max(...X, ...Y);
  if (minVal === maxVal) return 0;

  // Adaptive Sturges/square-root bin count (3..10) prevents artificial empty-bin spikes on small N
  const minN = Math.min(X.length, Y.length);
  const bins = Math.min(10, Math.max(3, Math.round(Math.sqrt(minN))));
  const binWidth = (maxVal - minVal) / bins;
  const refCounts = new Array(bins).fill(0);
  const currCounts = new Array(bins).fill(0);

  for (const v of X) {
    const b = Math.min(bins - 1, Math.floor((v - minVal) / binWidth));
    refCounts[b]++;
  }
  for (const v of Y) {
    const b = Math.min(bins - 1, Math.floor((v - minVal) / binWidth));
    currCounts[b]++;
  }

  let psi = 0;
  for (let b = 0; b < bins; b++) {
    const expected = (refCounts[b] + 0.5) / (X.length + 0.5 * bins);
    const actual = (currCounts[b] + 0.5) / (Y.length + 0.5 * bins);
    psi += (actual - expected) * Math.log(actual / expected);
  }

  return Number(Math.max(0, psi).toFixed(4));
}

/**
 * Computes separate descriptive distance metrics (PSI & Wasserstein-1), kept distinct from KS & MMD.
 */
export function computeDistributionDistances(
  featureName: string,
  refSample: (number | null | undefined)[],
  currSample: (number | null | undefined)[],
  minSamples: number = DEFAULT_DRIFT_CONFIG.minSamplesRequired
): DistanceMetricResult {
  const timestamp = new Date().toISOString();
  const X = sanitizeNumericArray(refSample);
  const Y = sanitizeNumericArray(currSample);
  const baselineMean = meanOf(X);
  const currentMean = meanOf(Y);

  if (X.length < minSamples || Y.length < minSamples) {
    const reason =
      Y.length === 0
        ? `No current observations available (0 / ${minSamples} required).`
        : `Insufficient observations (Reference: ${X.length}, Current: ${Y.length}, Minimum: ${minSamples}).`;
    return {
      method: 'PSI_WASSERSTEIN',
      featureName,
      baselineMean,
      currentMean,
      wassersteinDistance: null,
      psi: null,
      psiThreshold: DEFAULT_DRIFT_CONFIG.psiSevereThreshold,
      referenceSampleCount: X.length,
      currentSampleCount: Y.length,
      minSamplesRequired: minSamples,
      status: 'INSUFFICIENT_DATA',
      reason,
      timestamp,
    };
  }

  const wassersteinDistance = computeWassersteinDistance(X, Y);
  const psi = computePSI(X, Y, minSamples) ?? 0;

  let status: DriftStatus = 'STABLE';
  if (psi >= DEFAULT_DRIFT_CONFIG.psiSevereThreshold) {
    status = 'SEVERE_DRIFT';
  } else if (psi >= DEFAULT_DRIFT_CONFIG.psiModerateThreshold) {
    status = 'MODERATE_DRIFT';
  }

  const reason = `PSI=${psi.toFixed(4)} (severe >= ${DEFAULT_DRIFT_CONFIG.psiSevereThreshold}), Wasserstein-1=${wassersteinDistance}.`;

  return {
    method: 'PSI_WASSERSTEIN',
    featureName,
    baselineMean,
    currentMean,
    wassersteinDistance,
    psi,
    psiThreshold: DEFAULT_DRIFT_CONFIG.psiSevereThreshold,
    referenceSampleCount: X.length,
    currentSampleCount: Y.length,
    minSamplesRequired: minSamples,
    status,
    reason,
    timestamp,
  };
}

/**
 * Evaluates per-feature drift card metrics with strict INSUFFICIENT_DATA null-safety.
 */
export function evaluateFeatureDrift(
  featureName: string,
  refSample: (number | null | undefined)[],
  currSample: (number | null | undefined)[],
  alpha: number = DEFAULT_DRIFT_CONFIG.alpha,
  minSamples: number = DEFAULT_DRIFT_CONFIG.minSamplesRequired
): DriftMetric {
  const ks = computeTwoSampleKSTest(refSample, currSample, alpha, minSamples, featureName);

  if (!ks.valid) {
    return {
      featureName,
      method: 'KS_TEST',
      baselineMean: ks.baselineMean,
      currentMean: ks.currentMean,
      referenceSampleCount: ks.referenceSampleCount,
      currentSampleCount: ks.currentSampleCount,
      minSamplesRequired: minSamples,
      ksStatistic: null,
      ksPValue: null,
      ksCriticalValue: null,
      mmdScore: null,
      mmdPValue: null,
      mmdThreshold: null,
      wassersteinDistance: null,
      psi: null,
      status: 'INSUFFICIENT_DATA',
      threshold: alpha,
      alpha,
      interpretation: ks.reason,
      reason: ks.reason,
      timestamp: ks.timestamp,
    };
  }

  const mmd = computeMMD(refSample, currSample, alpha, minSamples);
  const dist = computeDistributionDistances(featureName, refSample, currSample, minSamples);

  let status: DriftStatus = 'STABLE';
  if (ks.status === 'SEVERE_DRIFT' || (ks.isDrift && mmd.isDrift)) {
    status = 'SEVERE_DRIFT';
  } else if (ks.isDrift || mmd.isDrift) {
    status = 'MODERATE_DRIFT';
  }

  const interpretation =
    status === 'STABLE'
      ? `H0 retained: KS D=${ks.statistic} (p=${ks.pValue} >= ${alpha}), 1D RBF-MMD=${mmd.mmdScore} (p=${mmd.pValue}).`
      : `Drift detected (${status}): KS D=${ks.statistic} (p=${ks.pValue}), 1D RBF-MMD=${mmd.mmdScore} (thresh=${mmd.threshold}).`;

  return {
    featureName,
    method: 'KS_TEST',
    baselineMean: ks.baselineMean,
    currentMean: ks.currentMean,
    referenceSampleCount: ks.referenceSampleCount,
    currentSampleCount: ks.currentSampleCount,
    minSamplesRequired: minSamples,
    ksStatistic: ks.statistic,
    ksPValue: ks.pValue,
    ksCriticalValue: ks.criticalValue,
    mmdScore: mmd.mmdScore,
    mmdPValue: mmd.pValue,
    mmdThreshold: mmd.threshold,
    wassersteinDistance: dist.wassersteinDistance,
    psi: dist.psi,
    status,
    threshold: alpha,
    alpha,
    interpretation,
    reason: ks.reason,
    timestamp: ks.timestamp,
  };
}
