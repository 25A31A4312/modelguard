export interface SampleImagePreset {
  id: string;
  url: string;
  name: string;
  category: string;
  description: string;
}

export const SAMPLE_PRESETS: SampleImagePreset[] = [
  {
    id: 'daylight-sports-car',
    url: '/samples/daylight-sports-car.jpg',
    name: 'Daylight Sports Car',
    category: 'Reference',
    description: 'Clear daylight image for baseline comparison.',
  },
  {
    id: 'foggy-highway',
    url: '/samples/foggy-highway.jpg',
    name: 'Foggy Highway',
    category: 'Weather Drift',
    description: 'Foggy conditions for testing image quality drift.',
  },
  {
    id: 'night-highway',
    url: '/samples/night-highway.jpg',
    name: 'Night Highway',
    category: 'Low Light',
    description: 'Low-light image for brightness and visibility testing.',
  },
  {
    id: 'urban-traffic',
    url: '/samples/urban-traffic.jpg',
    name: 'Urban Traffic',
    category: 'Traffic',
    description: 'Urban scene for traffic image testing.',
  },
];