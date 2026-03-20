import React, { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  CircularProgress,
  Container,
  Divider,
  FormControl,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableRow,
  Typography,
} from '@mui/material';
import { useSearchParams } from 'react-router-dom';
import { fetchRunDetails, fetchRuns, getLatestRunsByModel } from '../api';
import { MODEL_OPTIONS, RunDetails } from '../models';

const LABEL_BY_KEY: Record<string, string> = {
  accuracy: 'Accuracy',
  precision: 'Precision',
  recall: 'Recall',
  f1_score: 'F1 Score',
};

const EvaluationPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const runIdFromQuery = searchParams.get('runId');
  const modelIdFromQuery = searchParams.get('modelId');

  const [selectedModelId, setSelectedModelId] = useState<string>(
    modelIdFromQuery && MODEL_OPTIONS.some((m) => m.id === modelIdFromQuery)
      ? modelIdFromQuery
      : MODEL_OPTIONS[0].id,
  );
  const [runId, setRunId] = useState<string | null>(runIdFromQuery);
  const [runDetails, setRunDetails] = useState<RunDetails | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const selectedModel = useMemo(
    () => MODEL_OPTIONS.find((m) => m.id === selectedModelId) || MODEL_OPTIONS[0],
    [selectedModelId],
  );

  useEffect(() => {
    if (modelIdFromQuery && MODEL_OPTIONS.some((m) => m.id === modelIdFromQuery)) {
      setSelectedModelId(modelIdFromQuery);
    }
  }, [modelIdFromQuery]);

  useEffect(() => {
    const resolveRunIdAndLoadMetrics = async () => {
      setLoading(true);
      setError(null);
      setRunDetails(null);

      try {
        let resolvedRunId: string | null = null;

        if (runIdFromQuery && modelIdFromQuery === selectedModelId) {
          resolvedRunId = runIdFromQuery;
        }

        if (!resolvedRunId) {
          const runs = await fetchRuns();
          const latestRuns = getLatestRunsByModel(runs, MODEL_OPTIONS);
          resolvedRunId = latestRuns[selectedModelId]?.run_id || null;
        }

        if (!resolvedRunId) {
          setError('No run is available for this model yet.');
          setRunId(null);
          return;
        }

        setRunId(resolvedRunId);
        const details = await fetchRunDetails(resolvedRunId);
        setRunDetails(details);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Failed to load evaluation metrics.';
        setError(message);
      } finally {
        setLoading(false);
      }
    };

    void resolveRunIdAndLoadMetrics();
  }, [selectedModelId, runIdFromQuery, modelIdFromQuery]);

  const metrics = runDetails?.metrics || {};
  const primaryMetrics = Object.keys(LABEL_BY_KEY)
    .filter((key) => metrics[key] !== undefined)
    .map((key) => ({ key, label: LABEL_BY_KEY[key], value: metrics[key] }));

  const additionalMetrics = Object.entries(metrics).filter(([key]) => !(key in LABEL_BY_KEY));

  const trainingPlotUrl = runDetails?.plots?.training_progress;
  const confusionMatrixUrl = runDetails?.plots?.confusion_matrix;

  const handleModelChange = (nextModelId: string) => {
    setSelectedModelId(nextModelId);
    setSearchParams({ modelId: nextModelId });
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Paper sx={{ p: 3, borderRadius: 3 }}>
        <Typography variant="h4" fontWeight={700} gutterBottom>
          Evaluation
        </Typography>

        <Stack spacing={2} sx={{ mb: 2 }}>
          <FormControl fullWidth>
            <InputLabel id="evaluation-model-select-label">Model</InputLabel>
            <Select
              labelId="evaluation-model-select-label"
              value={selectedModelId}
              label="Model"
              onChange={(event) => handleModelChange(event.target.value)}
            >
              {MODEL_OPTIONS.map((model) => (
                <MenuItem key={model.id} value={model.id}>
                  {model.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Typography color="text.secondary">Model: {selectedModel.label}</Typography>
          {runId && (
            <Typography variant="body2" color="text.secondary">
              Run ID: {runId}
            </Typography>
          )}
        </Stack>

        <Divider sx={{ my: 2 }} />

        {loading && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <CircularProgress size={20} />
            <Typography>Loading evaluation metrics...</Typography>
          </Box>
        )}

        {error && <Alert severity="warning">{error}</Alert>}

        {!loading && !error && (
          <Stack spacing={3}>
            {primaryMetrics.length > 0 ? (
              <Table size="small">
                <TableBody>
                  {primaryMetrics.map((metric) => (
                    <TableRow key={metric.key}>
                      <TableCell>{metric.label}</TableCell>
                      <TableCell>
                        {typeof metric.value === 'number' ? metric.value.toFixed(4) : String(metric.value)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <Typography color="text.secondary">
                No standard metrics (accuracy, precision, recall, F1) were found in this run.
              </Typography>
            )}

            {additionalMetrics.length > 0 && (
              <>
                <Typography variant="h6">Additional Metrics</Typography>
                <Table size="small">
                  <TableBody>
                    {additionalMetrics.map(([key, value]) => (
                      <TableRow key={key}>
                        <TableCell>{key}</TableCell>
                        <TableCell>{typeof value === 'number' ? value.toFixed(4) : String(value)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </>
            )}

            <Divider />

            <Stack spacing={2}>
              <Typography variant="h6">Training Progress</Typography>
              {trainingPlotUrl ? (
                <Box
                  component="img"
                  src={trainingPlotUrl}
                  alt="Training progress"
                  sx={{ width: '100%', borderRadius: 2, border: '1px solid #e0e0e0' }}
                />
              ) : (
                <Typography color="text.secondary">Training plot not generated yet.</Typography>
              )}
            </Stack>

            <Stack spacing={2}>
              <Typography variant="h6">Confusion Matrix</Typography>
              {confusionMatrixUrl ? (
                <Box
                  component="img"
                  src={confusionMatrixUrl}
                  alt="Confusion matrix"
                  sx={{ width: '100%', borderRadius: 2, border: '1px solid #e0e0e0' }}
                />
              ) : (
                <Typography color="text.secondary">Confusion matrix not generated yet.</Typography>
              )}
            </Stack>
          </Stack>
        )}
      </Paper>
    </Container>
  );
};

export default EvaluationPage;
