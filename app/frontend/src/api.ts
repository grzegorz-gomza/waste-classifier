import axios from 'axios';
import { ModelOption, PredictionResponse, RunDetails, RunSummary } from './models';

export async function fetchRuns(): Promise<RunSummary[]> {
  const { data } = await axios.get<RunSummary[]>('/api/runs/');
  return data;
}

export function getLatestRunsByModel(
  runs: RunSummary[],
  models: ModelOption[],
): Record<string, RunSummary | undefined> {
  const sortedRuns = [...runs].sort((a, b) => b.start_time - a.start_time);

  return models.reduce<Record<string, RunSummary | undefined>>((acc, model) => {
    acc[model.id] = sortedRuns.find(
      (run) =>
        run.status === 'FINISHED' &&
        run.model_type === model.backendModelType &&
        run.model_name === model.backendModelName,
    );
    return acc;
  }, {});
}

export async function predictWithRun(
  runId: string,
  modelType: 'DL' | 'ML',
  file: File,
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('run_id', runId);
  formData.append('file', file);

  const endpoint = modelType === 'DL' ? '/api/predict/dl' : '/api/predict/ml';
  const { data } = await axios.post<PredictionResponse>(endpoint, formData);
  return data;
}

export async function fetchRunDetails(runId: string): Promise<RunDetails> {
  const { data } = await axios.get<RunDetails>(`/api/runs/${runId}`);
  return data;
}
