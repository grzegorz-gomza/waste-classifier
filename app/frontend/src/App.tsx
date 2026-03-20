import React from 'react';
import { AppBar, Box, Button, Container, CssBaseline, Toolbar, Typography } from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import RecyclingIcon from '@mui/icons-material/Recycling';
import { BrowserRouter as Router, Link, Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage';
import ResultsPage from './pages/ResultsPage';
import EvaluationPage from './pages/EvaluationPage';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#0B5FFF' },
    secondary: { main: '#189A5B' },
    background: { default: '#f3f7fb' },
  },
  shape: { borderRadius: 12 },
  typography: {
    fontFamily: '"Segoe UI", "Arial", sans-serif',
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ minHeight: '100vh' }}>
          <AppBar position="static" color="inherit" elevation={0} sx={{ borderBottom: '1px solid #e3e9f2' }}>
            <Container maxWidth="lg">
              <Toolbar disableGutters sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <RecyclingIcon color="primary" />
                  <Typography variant="h6" fontWeight={700}>Waste Classifier Demo</Typography>
                </Box>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button component={Link} to="/">Home</Button>
                  <Button component={Link} to="/evaluation">Evaluation</Button>
                </Box>
              </Toolbar>
            </Container>
          </AppBar>

          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/evaluation" element={<EvaluationPage />} />
          </Routes>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
