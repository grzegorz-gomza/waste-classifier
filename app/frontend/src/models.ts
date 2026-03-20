export type BackendModelType = 'DL' | 'ML';

export interface ModelOption {
  id: string;
  label: string;
  backendModelName: string;
  backendModelType: BackendModelType;
}

export interface RunSummary {
  run_id: string;
  run_name: string;
  model_type: string;
  model_name: string;
  status: string;
  start_time: number;
  end_time: number | null;
  artifact_uri: string;
}

export interface PredictionResponse {
  run_id: string;
  model_type: BackendModelType;
  class: string;
  confidence: number;
  probabilities?: Record<string, number>;
  // Enhanced prediction information from backend
  prediction?: {
    class: string;
    confidence: number;
    class_index: number;
    total_classes: number;
    description: string;
  };
}

export interface RunDetails extends RunSummary {
  params: Record<string, string>;
  metrics: Record<string, number>;
  plots?: {
    training_progress?: string;
    confusion_matrix?: string;
  };
  tags: Record<string, string>;
}

export const MODEL_OPTIONS: ModelOption[] = [
  {
    id: 'cnn-model-a',
    label: 'CNN Model A',
    backendModelName: 'resnet50',
    backendModelType: 'DL',
  },
  {
    id: 'cnn-model-b',
    label: 'CNN Model B',
    backendModelName: 'mobilenet_v2',
    backendModelType: 'DL',
  },
  {
    id: 'cnn-model-c',
    label: 'CNN Model C',
    backendModelName: 'efficientnet_b0',
    backendModelType: 'DL',
  },
  {
    id: 'random-forest-model',
    label: 'Random Forest Model',
    backendModelName: 'xgboost',
    backendModelType: 'ML',
  },
];

export const SAMPLE_IMAGES = [
  '/samples/sample1.png',
  '/samples/sample2.png',
  '/samples/sample3.png',
  '/samples/sample4.png',
  '/samples/sample5.png',
  '/samples/sample6.png',
];
