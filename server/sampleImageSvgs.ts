/**
 * Reliable, self-contained vector image representations for test and review samples.
 * Prevents broken image icons caused by remote CDN blocks, CORS, or truncated base64 strings.
 */

function svgToDataUri(svgString: string): string {
  const cleaned = svgString.replace(/\s+/g, ' ').trim();
  return `data:image/svg+xml;utf8,${encodeURIComponent(cleaned)}`;
}

export const FOGGY_HIGHWAY_DATA_URI = svgToDataUri(`
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500" width="800" height="500">
  <defs>
    <linearGradient id="fogSky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#94a3b8" />
      <stop offset="50%" stop-color="#cbd5e1" />
      <stop offset="100%" stop-color="#e2e8f0" />
    </linearGradient>
    <linearGradient id="road" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#64748b" />
      <stop offset="100%" stop-color="#334155" />
    </linearGradient>
    <filter id="fogBlur">
      <feGaussianBlur stdDeviation="8" />
    </filter>
  </defs>
  <!-- Background with dense fog -->
  <rect width="800" height="500" fill="url(#fogSky)" />
  <!-- Road perspective -->
  <polygon points="360,240 440,240 760,500 40,500" fill="url(#road)" />
  <!-- Road lanes -->
  <line x1="400" y1="240" x2="400" y2="500" stroke="#f1f5f9" stroke-width="4" stroke-dasharray="25,25" />
  <!-- Distant vehicle silhouette in fog -->
  <g opacity="0.65" filter="url(#fogBlur)">
    <rect x="375" y="250" width="50" height="30" rx="6" fill="#1e293b" />
    <!-- Red tail lights -->
    <circle cx="383" cy="272" r="4" fill="#ef4444" />
    <circle cx="417" cy="272" r="4" fill="#ef4444" />
  </g>
  <!-- Closer vehicle emerging from fog -->
  <g opacity="0.88">
    <rect x="330" y="320" width="140" height="80" rx="12" fill="#0f172a" />
    <!-- Rear windshield -->
    <polygon points="345,330 455,330 445,360 355,360" fill="#475569" />
    <!-- Glowing tail lights -->
    <circle cx="348" cy="378" r="8" fill="#ef4444" filter="drop-shadow(0 0 8px #ef4444)" />
    <circle cx="452" cy="378" r="8" fill="#ef4444" filter="drop-shadow(0 0 8px #ef4444)" />
  </g>
  <!-- Fog overlay bands -->
  <rect x="0" y="200" width="800" height="150" fill="#ffffff" opacity="0.45" filter="url(#fogBlur)" />
  <text x="30" y="45" font-family="sans-serif" font-size="18" font-weight="bold" fill="#0f172a">
    Foggy Highway — Adverse Weather Drift Test Sample
  </text>
  <text x="30" y="70" font-family="sans-serif" font-size="13" fill="#334155">
    Severe edge gradient attenuation (Sharpness: 24.8/100)
  </text>
</svg>
`);

export const SPORTS_CAR_DATA_URI = svgToDataUri(`
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500" width="800" height="500">
  <defs>
    <linearGradient id="skyDay" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#38bdf8" />
      <stop offset="60%" stop-color="#bae6fd" />
      <stop offset="100%" stop-color="#f0f9ff" />
    </linearGradient>
    <linearGradient id="carBody" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#dc2626" />
      <stop offset="50%" stop-color="#ef4444" />
      <stop offset="100%" stop-color="#b91c1c" />
    </linearGradient>
  </defs>
  <rect width="800" height="500" fill="url(#skyDay)" />
  <!-- Asphalt Ground -->
  <rect x="0" y="320" width="800" height="180" fill="#1e293b" />
  <line x1="0" y1="410" x2="800" y2="410" stroke="#fbbf24" stroke-width="4" stroke-dasharray="40,30" />
  <!-- Sports Car Silhouette -->
  <g>
    <!-- Shadow -->
    <ellipse cx="400" cy="385" rx="260" ry="20" fill="#090d16" opacity="0.6" />
    <!-- Lower Body -->
    <path d="M 170 360 C 200 320, 260 315, 340 315 L 560 315 C 620 315, 660 340, 670 360 L 640 375 L 190 375 Z" fill="url(#carBody)" />
    <!-- Cabin -->
    <path d="M 280 315 C 330 250, 420 240, 520 240 C 560 240, 590 280, 605 315 Z" fill="#0f172a" />
    <!-- Windshield glass -->
    <path d="M 310 310 C 350 260, 420 250, 500 250 L 515 310 Z" fill="#94a3b8" opacity="0.8" />
    <!-- Wheels -->
    <circle cx="260" cy="370" r="34" fill="#0f172a" stroke="#64748b" stroke-width="6" />
    <circle cx="260" cy="370" r="14" fill="#cbd5e1" />
    <circle cx="560" cy="370" r="34" fill="#0f172a" stroke="#64748b" stroke-width="6" />
    <circle cx="560" cy="370" r="14" fill="#cbd5e1" />
  </g>
  <text x="30" y="45" font-family="sans-serif" font-size="18" font-weight="bold" fill="#0369a1">
    Daylight Sports Car — High Confidence ImageNet Reference
  </text>
  <text x="30" y="70" font-family="sans-serif" font-size="13" fill="#0284c7">
    Correct ResNet-18 ImageNet Mapping (index 817: sports car, 94.2% confidence)
  </text>
</svg>
`);

export const URBAN_TRAFFIC_DATA_URI = svgToDataUri(`
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500" width="800" height="500">
  <defs>
    <linearGradient id="citySky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0284c7" />
      <stop offset="100%" stop-color="#e0f2fe" />
    </linearGradient>
  </defs>
  <rect width="800" height="500" fill="url(#citySky)" />
  <rect x="50" y="120" width="100" height="230" fill="#334155" />
  <rect x="180" y="80" width="120" height="270" fill="#1e293b" />
  <rect x="330" y="140" width="140" height="210" fill="#475569" />
  <rect x="500" y="90" width="110" height="260" fill="#334155" />
  <rect x="640" y="130" width="120" height="220" fill="#1e293b" />
  <rect x="0" y="350" width="800" height="150" fill="#0f172a" />
  <line x1="0" y1="425" x2="800" y2="425" stroke="#f8fafc" stroke-width="4" stroke-dasharray="30,25" />
  <rect x="680" y="220" width="12" height="150" fill="#64748b" />
  <rect x="668" y="160" width="36" height="70" rx="6" fill="#020617" />
  <circle cx="686" cy="175" r="8" fill="#ef4444" />
  <circle cx="686" cy="195" r="8" fill="#f59e0b" opacity="0.3" />
  <circle cx="686" cy="215" r="8" fill="#10b981" opacity="0.3" />
  <rect x="220" y="380" width="130" height="55" rx="8" fill="#2563eb" />
  <rect x="420" y="340" width="180" height="95" rx="6" fill="#f59e0b" />
  <text x="30" y="45" font-family="sans-serif" font-size="18" font-weight="bold" fill="#0f172a">
    Urban Traffic &amp; Intersection — Multi-Object Detection
  </text>
  <text x="30" y="70" font-family="sans-serif" font-size="13" fill="#334155">
    Objects: Cars, Buses, Traffic Signals, Pedestrians
  </text>
</svg>
`);

export const NIGHT_HIGHWAY_DATA_URI = svgToDataUri(`
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500" width="800" height="500">
  <defs>
    <linearGradient id="nightSky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#020617" />
      <stop offset="100%" stop-color="#0f172a" />
    </linearGradient>
  </defs>
  <rect width="800" height="500" fill="url(#nightSky)" />
  <polygon points="370,260 430,260 780,500 20,500" fill="#090d16" />
  <line x1="400" y1="260" x2="400" y2="500" stroke="#38bdf8" stroke-width="2" opacity="0.6" stroke-dasharray="20,20" />
  <circle cx="360" cy="340" r="12" fill="#ef4444" filter="drop-shadow(0 0 16px #ef4444)" />
  <circle cx="440" cy="340" r="12" fill="#ef4444" filter="drop-shadow(0 0 16px #ef4444)" />
  <text x="30" y="45" font-family="sans-serif" font-size="18" font-weight="bold" fill="#e2e8f0">
    Rainy Highway at Dusk — Low-Light Drift
  </text>
  <text x="30" y="70" font-family="sans-serif" font-size="13" fill="#94a3b8">
    Luminance drop (Brightness &lt; 80), wet asphalt specular reflections
  </text>
</svg>
`);
