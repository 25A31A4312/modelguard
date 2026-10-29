/**
 * ImageNet-1k Reference Class Index Mapping (Selected Domain Classes)
 * In PyTorch torchvision models, class indices range from 0 to 999.
 * Bug explanation:
 * In naive implementations, class 0-indexed vs 1-indexed off-by-one errors or passing
 * raw logit indices into wrong lookup arrays caused car images (indices 436, 511, 751, 817, 829)
 * to be mismapped to index 69 (trilobite), 0 (tench), or 122.
 */

export const IMAGENET_CATEGORIES: Record<number, string> = {
  // Road & Transport Vehicles (Standard torchvision ImageNet indices)
  407: 'ambulance',
  436: 'beach wagon, station wagon, estate car',
  468: 'cab, hack, taxi',
  511: 'convertible',
  555: 'fire engine, fire truck',
  569: 'garbage truck, dustcart',
  575: 'golfcart, golf cart',
  609: 'jeep, landrover',
  627: 'limousine, limo',
  654: 'minibus',
  656: 'minivan',
  670: 'motor scooter, scooter',
  671: 'mountain bike, all-terrain bike, off-roader',
  705: 'passenger car, coach, carriage',
  751: 'racer, race car, racing car',
  817: 'sports car, sport car',
  829: 'streetcar, tram, tramcar, trolley, trolley car',
  864: 'tow truck, tow car, wrecker',
  867: 'trailer truck, tractor trailer, articulated lorry',
  874: 'tricycle',
  919: 'street sign',
  920: 'traffic light, traffic signal, stoplight',
  // Comparison index that was erroneously triggered by off-by-index bug
  69: 'trilobite (marine arthropod fossil)',
  0: 'tench (freshwater fish)',
  1: 'goldfish',
};

export const TRAFFIC_DETECTION_LABELS = [
  'car',
  'truck',
  'bus',
  'pedestrian',
  'motorcycle',
  'bicycle',
  'traffic light',
  'stop sign',
  'traffic cone',
  'road sign',
] as const;
