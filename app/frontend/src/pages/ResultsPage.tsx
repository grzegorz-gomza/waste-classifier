import React from 'react';
import {
  Box,
  Button,
  Container,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableRow,
  Typography,
  Chip,
} from '@mui/material';
import { Link, useLocation, useSearchParams } from 'react-router-dom';
import { PredictionResponse } from '../models';

// Helper function to extract class name from sample image path
const getGroundTruthClass = (previewUrl?: string | null): string | null => {
  if (!previewUrl || !previewUrl.includes('/samples/')) {
    return null;
  }
  
  // Extract class name from path like "/samples/aerosol_cans_sample.png"
  const match = previewUrl.match(/\/samples\/([^_]+)_sample\.png/);
  return match ? match[1] : null;
};

interface ResultLocationState {
  prediction?: PredictionResponse;
  previewUrl?: string | null;
  modelLabel?: string;
}

const ResultsPage: React.FC = () => {
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const runId = searchParams.get('runId') || '';
  const modelId = searchParams.get('modelId') || '';
  const state = (location.state || {}) as ResultLocationState;

  if (!state.prediction) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Paper sx={{ p: 3, borderRadius: 3 }}>
          <Typography variant="h6" gutterBottom>
            No prediction result in this session.
          </Typography>
          <Typography color="text.secondary" gutterBottom>
            Run a prediction from the Home page first.
          </Typography>
          <Button component={Link} to="/" variant="contained">Go To Home</Button>
        </Paper>
      </Container>
    );
  }

  const prediction = state.prediction;
  const groundTruthClass = getGroundTruthClass(state.previewUrl);
  const isCorrect = groundTruthClass && groundTruthClass === prediction.class;

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Stack spacing={3}>
        <Typography variant="h4" fontWeight={700}>Results</Typography>

        <Stack direction={{ xs: 'column', md: 'row' }} spacing={3}>
          <Paper sx={{ p: 3, borderRadius: 3, flex: 1 }}>
            <Typography variant="h6" gutterBottom>Image</Typography>
            {state.previewUrl ? (
              <Box
                component="img"
                src={state.previewUrl}
                alt="Prediction input"
                sx={{ width: '100%', maxHeight: 360, objectFit: 'contain', borderRadius: 2, border: '1px solid #e0e0e0' }}
              />
            ) : (
              <Typography color="text.secondary">Preview not available.</Typography>
            )}
          </Paper>

          <Paper sx={{ p: 3, borderRadius: 3, flex: 1 }}>
            <Typography variant="h6" gutterBottom>Prediction Output</Typography>
            <Table size="small">
              <TableBody>
                <TableRow>
                  <TableCell>Model</TableCell>
                  <TableCell>{state.modelLabel || modelId || '-'}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Predicted Class</TableCell>
                  <TableCell>
                    <Stack direction="row" alignItems="center" spacing={1}>
                      <span>{prediction.class}</span>
                      {groundTruthClass && (
                        <Chip
                          label={isCorrect ? "Correct" : "Incorrect"}
                          color={isCorrect ? "success" : "error"}
                          size="small"
                        />
                      )}
                    </Stack>
                  </TableCell>
                </TableRow>
                {groundTruthClass && (
                  <TableRow>
                    <TableCell>Ground Truth Class</TableCell>
                    <TableCell>{groundTruthClass}</TableCell>
                  </TableRow>
                )}
                <TableRow>
                  <TableCell>Confidence</TableCell>
                  <TableCell>{(prediction.confidence * 100).toFixed(2)}%</TableCell>
                </TableRow>
                {prediction.prediction?.description && (
                  <TableRow>
                    <TableCell sx={{ verticalAlign: 'top' }}>Class Description</TableCell>
                    <TableCell sx={{ wordBreak: 'break-word' }}>
                      {prediction.prediction.description}
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>

            {prediction.probabilities && Object.keys(prediction.probabilities).length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Typography fontWeight={600} gutterBottom>Class Probabilities</Typography>
                <Table size="small">
                  <TableBody>
                    {Object.entries(prediction.probabilities)
                      .sort((a, b) => b[1] - a[1])
                      .map(([className, probability]) => (
                        <TableRow key={className}>
                          <TableCell>{className}</TableCell>
                          <TableCell>{(probability * 100).toFixed(2)}%</TableCell>
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </Box>
            )}

            <Stack direction="row" spacing={1.5} sx={{ mt: 3 }}>
              <Button component={Link} to="/" variant="outlined">
                New Prediction
              </Button>
              <Button
                component={Link}
                to={`/evaluation?runId=${encodeURIComponent(runId)}&modelId=${encodeURIComponent(modelId)}`}
                variant="contained"
              >
                View Evaluation
              </Button>
            </Stack>
          </Paper>
        </Stack>
      </Stack>
    </Container>
  );
};

export default ResultsPage;
