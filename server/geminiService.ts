import { GoogleGenAI, Type } from '@google/genai';
import type { ImageQualityMetrics, InferenceResult, ModelType } from '../src/types/modelguard.ts';

const GEMINI_VISION_MODEL = 'gemini-3.1-flash-lite';

let aiInstance: GoogleGenAI | null = null;

function getAiClient(): GoogleGenAI | null {
  const key = process.env.GEMINI_API_KEY?.trim();
  if (!key || key === 'MY_GEMINI_API_KEY') {
    return null;
  }
  if (!aiInstance) {
    aiInstance = new GoogleGenAI({
      apiKey: key,
      httpOptions: {
        timeout: 15000,
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return aiInstance;
}

const normalizeCoord = (c: unknown): number => {
  const val = Number(c) || 0;
  if (val > 1.0) {
    return Math.min(1.0, Math.max(0.0, Number((val / 1000).toFixed(4))));
  }
  return Math.min(1.0, Math.max(0.0, Number(val.toFixed(4))));
};

export function getConfiguredGeminiModel(): string {
  return GEMINI_VISION_MODEL;
}

/**
 * Executes multimodal vision classification or object detection via Gemini API,
 * while strictly pairing the prediction with deterministic pixel-computed qualityMetrics.
 */
export async function runServerInference(
  imageBase64: string,
  mimeType: string = 'image/jpeg',
  modelType: ModelType = 'resnet18_classifier',
  pixelQualityMetrics: ImageQualityMetrics
): Promise<InferenceResult | null> {
  const ai = getAiClient();
  if (!ai) {
    return null;
  }

  const cleanBase64 = imageBase64.replace(/^data:image\/[a-zA-Z0-9+.-]+;base64,/, '');
  const startTime = Date.now();

  try {
    if (modelType === 'traffic_object_detector') {
      const prompt = `Analyze this road or traffic image as an object detector.
Identify visible traffic actors and objects (e.g., car, truck, bus, pedestrian, motorcycle, bicycle, traffic light, road sign).
For each detected object return:
- label
- confidence between 0.0 and 1.0
- normalized bounding box coordinates strictly in [0.0, 1.0]: ymin, xmin, ymax, xmax.`;

      const response = await ai.models.generateContent({
        model: GEMINI_VISION_MODEL,
        contents: [
          {
            role: 'user',
            parts: [
              {
                inlineData: {
                  mimeType,
                  data: cleanBase64,
                },
              },
              { text: prompt },
            ],
          },
        ],
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              sceneDescription: { type: Type.STRING },
              primaryCategory: { type: Type.STRING },
              objects: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    label: { type: Type.STRING },
                    confidence: { type: Type.NUMBER },
                    ymin: { type: Type.NUMBER },
                    xmin: { type: Type.NUMBER },
                    ymax: { type: Type.NUMBER },
                    xmax: { type: Type.NUMBER },
                  },
                  required: ['label', 'confidence', 'ymin', 'xmin', 'ymax', 'xmax'],
                },
              },
            },
            required: ['sceneDescription', 'primaryCategory', 'objects'],
          },
        },
      });

      const parsed = JSON.parse(response.text || '{}');
      const latency = Date.now() - startTime;

      const topObjects = (parsed.objects || []).map((o: any) => ({
        label: String(o.label).toLowerCase(),
        confidence: Math.min(0.99, Math.max(0.05, Number(o.confidence) || 0.5)),
        ymin: normalizeCoord(o.ymin),
        xmin: normalizeCoord(o.xmin),
        ymax: normalizeCoord(o.ymax),
        xmax: normalizeCoord(o.xmax),
      }));

      const topConfidence =
        topObjects.length > 0 ? topObjects[0].confidence : 0.45;
      const isFoggy =
        pixelQualityMetrics.fogProbability >= 0.45 || pixelQualityMetrics.sharpness < 26;

      return {
        id: `inf-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        timestamp: new Date().toISOString(),
        modelType: 'traffic_object_detector',
        modelVersion: 'v1.1.0-traffic-detector',
        primaryPrediction: {
          label: parsed.sceneDescription || 'Traffic scene',
          confidence: topConfidence,
          category: parsed.primaryCategory || 'Road Transportation',
        },
        topClasses: topObjects.slice(0, 5).map((o: any, idx: number) => ({
          className: o.label,
          confidence: o.confidence,
          classId: 100 + idx,
        })),
        detectedObjects: topObjects,
        qualityMetrics: pixelQualityMetrics,
        driftFlags: {
          isLowConfidence: topConfidence < 0.6,
          isDriftSuspect: isFoggy,
          reason: isFoggy
            ? `Covariate Drift: Pixel-measured haze ${(pixelQualityMetrics.fogProbability * 100).toFixed(0)}%, Laplacian sharpness ${pixelQualityMetrics.sharpness}.`
            : undefined,
        },
        latencyMs: latency,
      };
    }

    const prompt = `Classify the primary subject of this image using standard ImageNet-1k categories (e.g., sports car, convertible, station wagon, minivan, cab, trailer truck, street sign, traffic light, etc.).
Provide the top 5 predicted ImageNet classes with probabilities summing to approximately 1.0.`;

    const response = await ai.models.generateContent({
      model: GEMINI_VISION_MODEL,
      contents: [
        {
          role: 'user',
          parts: [
            {
              inlineData: {
                mimeType,
                data: cleanBase64,
              },
            },
            { text: prompt },
          ],
        },
      ],
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            primaryLabel: { type: Type.STRING },
            confidence: { type: Type.NUMBER },
            top5: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  className: { type: Type.STRING },
                  confidence: { type: Type.NUMBER },
                  classId: { type: Type.INTEGER },
                },
                required: ['className', 'confidence', 'classId'],
              },
            },
          },
          required: ['primaryLabel', 'confidence', 'top5'],
        },
      },
    });

    const parsed = JSON.parse(response.text || '{}');
    const latency = Date.now() - startTime;

    const topClasses = (parsed.top5 || []).map((c: any) => ({
      className: String(c.className),
      confidence: Math.min(0.999, Math.max(0.001, Number(c.confidence) || 0.1)),
      classId: Number(c.classId) || 817,
    }));

    const primaryConfidence = Math.min(
      0.999,
      Math.max(0.01, Number(parsed.confidence) || (topClasses[0]?.confidence ?? 0.5))
    );
    const isFoggy =
      pixelQualityMetrics.fogProbability >= 0.45 || pixelQualityMetrics.sharpness < 26;

    return {
      id: `inf-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      timestamp: new Date().toISOString(),
      modelType: 'resnet18_classifier',
      modelVersion: 'v1.0.0-resnet18',
      primaryPrediction: {
        label: parsed.primaryLabel || topClasses[0]?.className || 'motor vehicle',
        confidence: primaryConfidence,
        category: 'ImageNet-1k Classification',
      },
      topClasses,
      detectedObjects: [],
      qualityMetrics: pixelQualityMetrics,
      driftFlags: {
        isLowConfidence: primaryConfidence < 0.6,
        isDriftSuspect: isFoggy,
        reason: isFoggy
          ? `Covariate Drift: Pixel-measured haze ${(pixelQualityMetrics.fogProbability * 100).toFixed(0)}%, Laplacian sharpness ${pixelQualityMetrics.sharpness}.`
          : undefined,
      },
      latencyMs: latency,
    };
  } catch (err) {
    console.error('[ModelGuard Gemini Inference Error]', err);
    return null;
  }
}
