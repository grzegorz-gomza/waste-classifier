import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Button,
  Box,
} from '@mui/material';
import { Link } from 'react-router-dom';
import axios from 'axios';

interface Run {
  run_id: string;
  run_name: string;
  model_type: string;
  model_name: string;
  status: string;
  start_time: number;
  end_time: number | null;
  artifact_uri: string;
}

const RunsPage: React.FC = () => {
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get('/api/runs/')
      .then(res => {
        console.log('API Response:', res.data);
        setRuns(res.data);
      })
      .catch(err => {
        console.error('API Error:', err);
        // Show fallback data when API fails
        setRuns([{
          run_id: "074592b1ae8947e39d67797ecb3a9056",
          run_name: "XGBoost_Test",
          model_type: "ML",
          model_name: "xgboost",
          status: "FINISHED",
          start_time: 1741467800000,
          end_time: 1741467800000,
          artifact_uri: "/mlruns/1/074592b1ae8947e39d67797ecb3a9056/artifacts",
        }]);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Typography>Loading runs...</Typography>;

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>Experiment Runs</Typography>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Run Name</TableCell>
              <TableCell>Model Type</TableCell>
              <TableCell>Model</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Start</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {runs.map(run => (
              <TableRow key={run.run_id}>
                <TableCell>{run.run_name}</TableCell>
                <TableCell>{run.model_type}</TableCell>
                <TableCell>{run.model_name}</TableCell>
                <TableCell>{run.status}</TableCell>
                <TableCell>{new Date(run.start_time).toLocaleString()}</TableCell>
                <TableCell>
                  <Box display="flex" gap={1}>
                    <Button
                      component={Link}
                      to={`/artifacts/${run.run_id}`}
                      size="small"
                      variant="outlined"
                    >
                      Artifacts
                    </Button>
                    <Button
                      component={Link}
                      to={`/predict?run_id=${run.run_id}&model_type=${run.model_type}`}
                      size="small"
                      variant="contained"
                    >
                      Predict
                    </Button>
                  </Box>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Container>
  );
};

export default RunsPage;
