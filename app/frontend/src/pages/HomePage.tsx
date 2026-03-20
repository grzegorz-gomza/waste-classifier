import React, { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardActionArea,
  CardMedia,
  CircularProgress,
  Container,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
  Typography,
  Chip,
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { useNavigate } from 'react-router-dom';
import { fetchRuns, getLatestRunsByModel, predictWithRun } from '../api';
import { MODEL_OPTIONS, PredictionResponse, RunSummary, SAMPLE_IMAGES } from '../models';

// Helper function to extract class name from sample image path
const getGroundTruthClass = (samplePath?: string | null): string | null => {
  if (!samplePath || !samplePath.includes('/samples/')) {
    return null;
  }
  
  // Extract class name from path like "/samples/aerosol_cans_sample.png"
  const match = samplePath.match(/\/samples\/([^_]+)_sample\.png/);
  return match ? match[1] : null;
};

async function sampleToFile(samplePath: string): Promise<File> {
  const response = await fetch(samplePath);
  const blob = await response.blob();
  const fileName = samplePath.split('/').pop() || 'sample.png';
  return new File([blob], fileName, { type: blob.type || 'image/png' });
}

const HomePage: React.FC = () => {
  const navigate = useNavigate();

  const [selectedModelId, setSelectedModelId] = useState<string>(MODEL_OPTIONS[0].id);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [sampleImagePath, setSampleImagePath] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loadingRuns, setLoadingRuns] = useState<boolean>(true);
  const [runsError, setRunsError] = useState<string | null>(null);

  const [predicting, setPredicting] = useState<boolean>(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [lastPrediction, setLastPrediction] = useState<PredictionResponse | null>(null);
  const [lastPredictionGroundTruth, setLastPredictionGroundTruth] = useState<string | null>(null);

  useEffect(() => {
    const loadRuns = async () => {
      setLoadingRuns(true);
      setRunsError(null);
      try {
        const data = await fetchRuns();
        setRuns(data);
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : 'Failed to load models from backend.';
        setRunsError(message);
      } finally {
        setLoadingRuns(false);
      }
    };

    void loadRuns();
  }, []);

  useEffect(() => {
    if (!uploadedFile) {
      return undefined;
    }

    const objectUrl = URL.createObjectURL(uploadedFile);
    setPreviewUrl(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [uploadedFile]);

  useEffect(() => {
    if (sampleImagePath) {
      setPreviewUrl(sampleImagePath);
    }
  }, [sampleImagePath]);

  const selectedModel = useMemo(
    () => MODEL_OPTIONS.find((model) => model.id === selectedModelId) || MODEL_OPTIONS[0],
    [selectedModelId],
  );

  const latestRunsByModel = useMemo(
    () => getLatestRunsByModel(runs, MODEL_OPTIONS),
    [runs],
  );

  const selectedRun = latestRunsByModel[selectedModel.id];

  const canPredict = Boolean((uploadedFile || sampleImagePath) && selectedRun && !predicting);

  const handleUploadChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setUploadedFile(file);
    setSampleImagePath(null);
    setPredictionError(null);
  };

  const handleSampleSelect = (samplePath: string) => {
    setSampleImagePath(samplePath);
    setUploadedFile(null);
    setPredictionError(null);
  };

  const handlePredict = async () => {
    if (!selectedRun) {
      setPredictionError('No pretrained run is available for the selected model yet.');
      return;
    }

    setPredicting(true);
    setPredictionError(null);

    try {
      const imageFile = uploadedFile || (sampleImagePath ? await sampleToFile(sampleImagePath) : null);

      if (!imageFile) {
        setPredictionError('Select an image first.');
        return;
      }

      const prediction: PredictionResponse = await predictWithRun(
        selectedRun.run_id,
        selectedModel.backendModelType,
        imageFile,
      );

      // Store prediction result and ground truth for display
      setLastPrediction(prediction);
      setLastPredictionGroundTruth(getGroundTruthClass(sampleImagePath));

      const params = new URLSearchParams({
        runId: selectedRun.run_id,
        modelId: selectedModel.id,
      });

      navigate(`/results?${params.toString()}`, {
        state: {
          prediction,
          previewUrl: previewUrl || sampleImagePath,
          modelLabel: selectedModel.label,
        },
      });
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : 'Prediction failed.';
      setPredictionError(message);
    } finally {
      setPredicting(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Stack spacing={3}>
        <Paper elevation={0} sx={{ p: 3, borderRadius: 3, background: 'linear-gradient(120deg, #edf7ff 0%, #f6fff0 100%)' }}>
          <Typography variant="h4" fontWeight={700} gutterBottom>
            Pretrained Model Demo
          </Typography>
          <Typography color="text.secondary">
            Upload an image or pick a sample, choose a model, and run inference.
          </Typography>
        </Paper>

        {runsError && <Alert severity="warning">{runsError}</Alert>}

        <Grid container spacing={3}>
          <Grid item xs={12} md={7}>
            <Paper sx={{ p: 3, borderRadius: 3, height: '100%' }}>
              <Stack spacing={2}>
                <Typography variant="h6" fontWeight={600}>1) Select Image</Typography>

                <Button component="label" variant="outlined" startIcon={<UploadFileIcon />}>
                  Upload JPG / PNG
                  <input
                    hidden
                    type="file"
                    accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                    onChange={handleUploadChange}
                  />
                </Button>

                <Typography variant="body2" color="text.secondary">
                  Or choose a sample image:
                </Typography>

                <Grid container spacing={1.5}>
                  {SAMPLE_IMAGES.map((samplePath) => {
                    const isSelected = sampleImagePath === samplePath;
                    return (
                      <Grid item xs={4} sm={3} key={samplePath}>
                        <Card
                          sx={{
                            borderRadius: 2,
                            border: isSelected ? '2px solid #1976d2' : '1px solid #e0e0e0',
                          }}
                        >
                          <CardActionArea onClick={() => handleSampleSelect(samplePath)}>
                            <CardMedia component="img" image={samplePath} alt={samplePath} sx={{ height: 88, objectFit: 'cover' }} />
                          </CardActionArea>
                        </Card>
                      </Grid>
                    );
                  })}
                </Grid>

                {previewUrl && (
                  <Box>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      Preview
                    </Typography>
                    <Box
                      component="img"
                      src={previewUrl}
                      alt="Selected preview"
                      sx={{ width: '100%', maxHeight: 300, objectFit: 'contain', borderRadius: 2, border: '1px solid #e0e0e0' }}
                    />
                  </Box>
                )}
              </Stack>
            </Paper>
          </Grid>

          <Grid item xs={12} md={5}>
            <Paper sx={{ p: 3, borderRadius: 3, height: '100%' }}>
              <Stack spacing={2.5}>
                <Typography variant="h6" fontWeight={600}>2) Choose Model</Typography>

                <FormControl fullWidth>
                  <InputLabel id="model-select-label">Model</InputLabel>
                  <Select
                    labelId="model-select-label"
                    value={selectedModelId}
                    label="Model"
                    onChange={(event) => setSelectedModelId(event.target.value)}
                  >
                    {MODEL_OPTIONS.map((model) => (
                      <MenuItem key={model.id} value={model.id}>
                        {model.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                <Box sx={{ p: 2, borderRadius: 2, border: '1px solid #e0e0e0', backgroundColor: '#fafafa' }}>
                  <Typography fontWeight={600}>{selectedModel.label}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Backend model: {selectedModel.backendModelName}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Type: {selectedModel.backendModelType}
                  </Typography>
                  <Typography variant="body2" color={selectedRun ? 'success.main' : 'error.main'} sx={{ mt: 1 }}>
                    {loadingRuns
                      ? 'Loading model runs...'
                      : selectedRun
                        ? `Using latest run: ${selectedRun.run_name}`
                        : 'No finished run available for this model.'}
                  </Typography>
                </Box>

                {predictionError && <Alert severity="error">{predictionError}</Alert>}

                <Button
                  variant="contained"
                  size="large"
                  startIcon={predicting ? <CircularProgress size={16} color="inherit" /> : <AutoAwesomeIcon />}
                  disabled={!canPredict}
                  onClick={() => {
                    void handlePredict();
                  }}
                >
                  {predicting ? 'Predicting...' : 'Predict'}
                </Button>
              </Stack>
            </Paper>
          </Grid>
        </Grid>

        {/* Prediction Results Section */}
        {lastPrediction && (
          <Paper sx={{ p: 3, borderRadius: 3 }}>
            <Typography variant="h6" fontWeight={600} gutterBottom>
              Latest Prediction Result
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={8}>
                <Stack spacing={1}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Predicted Class:
                    </Typography>
                    <Stack direction="row" alignItems="center" spacing={1}>
                      <Typography variant="body1" fontWeight={600}>
                        {lastPrediction.class}
                      </Typography>
                      {lastPredictionGroundTruth && (
                        <Chip
                          label={lastPredictionGroundTruth === lastPrediction.class ? "Correct" : "Incorrect"}
                          color={lastPredictionGroundTruth === lastPrediction.class ? "success" : "error"}
                          size="small"
                        />
                      )}
                    </Stack>
                  </Box>
                  {lastPredictionGroundTruth && (
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Ground Truth Class:
                      </Typography>
                      <Typography variant="body1">
                        {lastPredictionGroundTruth}
                      </Typography>
                    </Box>
                  )}
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Confidence:
                    </Typography>
                    <Typography variant="body1">
                      {(lastPrediction.confidence * 100).toFixed(2)}%
                    </Typography>
                  </Box>
                  {lastPrediction.prediction?.description && (
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Class Description:
                      </Typography>
                      <Typography variant="body2" sx={{ wordBreak: 'break-word' }}>
                        {lastPrediction.prediction.description}
                      </Typography>
                    </Box>
                  )}
                </Stack>
              </Grid>
              <Grid item xs={12} md={4}>
                <Stack spacing={1}>
                  <Button
                    component="a"
                    href="#"
                    onClick={(e) => {
                      e.preventDefault();
                      const params = new URLSearchParams({
                        runId: lastPrediction.run_id,
                        modelId: selectedModel.id,
                      });
                      navigate(`/results?${params.toString()}`, {
                        state: {
                          prediction: lastPrediction,
                          previewUrl: previewUrl || sampleImagePath,
                          modelLabel: selectedModel.label,
                        },
                      });
                    }}
                    variant="outlined"
                    size="small"
                  >
                    View Full Results
                  </Button>
                </Stack>
              </Grid>
            </Grid>
          </Paper>
        )}
      </Stack>
    </Container>
  );
};

export default HomePage;
