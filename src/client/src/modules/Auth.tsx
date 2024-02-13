import React from 'react';
import { Link } from 'react-router-dom';
import { Button, Typography, Container, Box, ThemeProvider, createTheme } from '@mui/material';
import { styled, keyframes } from '@mui/system';
import logo from './logo.jpg';

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(-20px); }
  to { opacity: 1; transform: translateY(0); }
`;

const Logo = styled('img')({
  width: 150,
  height: 'auto',
  marginTop: '2rem',
  animation: `${fadeIn} 1s ease-out`,
});

// Define a custom animation for the buttons
const pulse = keyframes`
  from { transform: scale(1); }
  50% { transform: scale(1.05); }
  to { transform: scale(1); }
`;

const StyledButton = styled(Button)({
  margin: '1rem',
  animation: `${pulse} 2s infinite`,
  '&:hover': {
    animation: 'none',
  },
});

const theme = createTheme({
  palette: {
    primary: {
      main: '#00a0b2', 
    },
    secondary: {
      main: '#E6F7FF', 
    },
    background: {
      default: '#E6F7FF', 
    },
  },
  typography: {
    fontFamily: 'Arial, sans-serif',
    h4: {
      fontWeight: 'bold',
      color: '#00796B', 
      animation: `${fadeIn} 1s ease-out`,
    },
    h5: {
      color: '#005B6A', // A darker color from the logo for contrast
      animation: `${fadeIn} 1s ease-out 0.5s`,
    },
  },
});

function Auth() {
  return (
    <ThemeProvider theme={theme}>
      <Container component="main" maxWidth="false" sx={{ bgcolor: 'background.default' }}>
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '100vh',
          }}
        >
          <Logo src={logo} alt="MotionMingle Logo" />
          <Typography variant="h4" component="h1" gutterBottom>
            Motion Mingle
          </Typography>
          <Typography variant="h4" gutterBottom>
            Harmony in Motion, Unity in Practice
          </Typography>
          <StyledButton variant="contained" color="primary">
            <Link to="/instructor" style={{ color: '#FFF', textDecoration: 'none' }}>
              Instructor
            </Link>
          </StyledButton>
          <Typography variant="h5">or</Typography>
          <StyledButton variant="contained" color="primary">
            <Link to="/practitioner" style={{ color: '#FFF', textDecoration: 'none' }}>
              Practitioner
            </Link>
          </StyledButton>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default Auth;
