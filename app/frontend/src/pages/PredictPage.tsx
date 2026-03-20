import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Container,
  Typography,
  Box,
  Button,
  CircularProgress,
  Alert,
} from '@mui/material';
import axios from 'axios';

interface PredictionResult {
  run_id: string;
  model_type: string;
  class: string;
  confidence: number;
}

const PredictPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const runId = searchParams.get('run_id') || '';
  const modelType = searchParams.get('model_type') || '';
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!file || !runId || !modelType) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('run_id', runId);
    try {
      const endpoint = modelType === 'DL' ? '/api/predict/dl' : '/api/predict/ml';
      const res = await axios.post<PredictionResult>(endpoint, formData);
      setResult(res.data);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>Predict</Typography>
      <Typography variant="subtitle1">
        Run: {runId} ({modelType})
      </Typography>

      <Box mt={2}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
      </Box>

      <Box mt={2}>
        <Button
          variant="contained"
          onClick={handlePredict}
          disabled={!file || loading}
        >
          {loading ? <CircularProgress size={20} /> : 'Predict'}
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      {result && (
        <Box mt={2} p={2} border={1} borderColor="grey.300" borderRadius={1}>
          <Typography variant="h6">Result</Typography>
          <Typography>Class: {result.class}</Typography>
          <Typography>Confidence: {(result.confidence * 100).toFixed(2)}%</Typography>
        </Box>
      )}
    </Container>
  );
};

export default PredictPage;
