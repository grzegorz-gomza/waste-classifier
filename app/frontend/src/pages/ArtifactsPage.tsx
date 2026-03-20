import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Container,
  Typography,
  List,
  ListItem,
  ListItemText,
  Button,
  Box,
  CircularProgress,
} from '@mui/material';
import axios from 'axios';

interface Artifact {
  path: string;
  is_dir: boolean;
  size: number | null;
}

const ArtifactsPage: React.FC = () => {
  const { runId } = useParams<{ runId: string }>();
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!runId) return;
    axios.get(`/api/artifacts/${runId}/list`)
      .then(res => setArtifacts(res.data))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [runId]);

  const handleDownload = (path: string) => {
    window.open(`/api/artifacts/${runId}/download?path=${encodeURIComponent(path)}`, '_blank');
  };

  if (loading) return <CircularProgress />;

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>Artifacts for {runId}</Typography>
      <List>
        {artifacts.map((artifact, idx) => (
          <ListItem key={idx} divider>
            <ListItemText
              primary={artifact.path}
              secondary={artifact.is_dir ? 'Directory' : `File (${artifact.size} bytes)`}
            />
            <Button onClick={() => handleDownload(artifact.path)} size="small" variant="outlined">
              Download
            </Button>
          </ListItem>
        ))}
      </List>
    </Container>
  );
};

export default ArtifactsPage;
