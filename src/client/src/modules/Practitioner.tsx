import React, { useRef, useState, useEffect } from 'react';
import {
  SelectChangeEvent,
  Button,
  Typography,
  Box,
  Container,
  ThemeProvider,
  createTheme
} from '@mui/material';
import { connectAsConsumer, createPeerConnection } from './RTCControl';
import SelectAnnotation from './SelectAnnotation';
import logo from './logo.jpg';

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
});

function Practitioner() {
  const createConsumerPeerConnection = () => {
    const pc = createPeerConnection();
    pc.addEventListener('track', (event) => {
      if (event.track.kind === 'video') {
        const remoteVideo = remoteVideoRef.current;
        let rest;
        [remoteVideo!.srcObject, ...rest] = event.streams;
        console.log('remoteVideo');
      }
    });
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });
    return pc;
  };

  const remoteVideoRef = useRef<HTMLVideoElement>(null);
  const [consumerPc, setConsumerPc] = useState<RTCPeerConnection>(createConsumerPeerConnection());
  const [selectedAnnotation, setSelectedAnnotation] = useState<string>('');
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [showInstructions, setShowInstructions] = useState(false);

  const closeRemote = async () => {
    // consumerPc.close();
    setConsumerPc(createConsumerPeerConnection());
    remoteVideoRef.current!.srcObject = null;
    setIsConnected(false);
  };

  const consume = async () => {
    await connectAsConsumer(consumerPc, selectedAnnotation);
    remoteVideoRef.current?.play();
    setIsConnected(true);
  };

  const selectNewAnnotationAndRefreshVideo = (event: SelectChangeEvent) => {
    setSelectedAnnotation(event.target.value);
  };

  useEffect(() => {
    if(isConnected) {
      closeRemote();
      consume();  
    }
  }, [selectedAnnotation])

  return (
    <ThemeProvider theme={theme}>
    <Box sx={{ bgcolor: 'background.default', minHeight: '100vh' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 2 }}>
        <Box sx={{ flex: 1 }} /> {/* Empty box for spacing */}
        <Box sx={{ display: 'flex', justifyContent: 'center', flex: 1 }}>
          <img src={logo} alt="Motion Mingle Logo" style={{ height: 50 }} />
          <Typography variant="h4" sx={{ ml: 2 }}>
            Motion Mingle
          </Typography>
        </Box>
        <Box sx={{ flex: 1, display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="contained"
            color="secondary"
            size="small"
            onClick={() => setShowInstructions(true)}
          >
            Instructions
          </Button>
        </Box>
      </Box>
        <Container maxWidth="md" sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          <Typography variant="h5" gutterBottom>
            Practitioner
          </Typography>
          <SelectAnnotation selectedAnnotation={selectedAnnotation} selectionHandler={selectNewAnnotationAndRefreshVideo} />
          <video
            ref={remoteVideoRef}
            autoPlay
            style={{ width: '100%', maxWidth: '600px', aspectRatio: '600/450', border: '3px solid', borderColor: 'primary.main', borderRadius: '4px', marginTop: '20px' }}
            playsInline
          />
          {isConnected ? (
            <Button variant="contained" color="error" size="large" onClick={closeRemote}>
              Disconnect
            </Button>
          ) : (
            <Button variant="contained" color="primary" size="large" onClick={consume}>
              Connect
            </Button>
          )}
          {showInstructions && (
          <Box
            style={{
              position: 'fixed',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              backgroundColor: 'white',
              padding: '20px',
              zIndex: 1000,
              maxWidth: '900px',
              maxHeight: '600px',
              overflowY: 'auto',
              border: '2px solid #000',
            }}
          >
            <Typography variant="h6" style={{ fontSize: '30px', lineHeight: '2' }} gutterBottom>
              App Instructions
            </Typography>
            <Typography variant="body1" style={{ fontSize: '25px', lineHeight: '2' }} gutterBottom>
              To join a streaming session, click the
              <span>
                &nbsp;
                <Button variant="contained" color="primary" size="large">
                  Connect
                </Button>
                &nbsp;
              </span>
              button.<br />
              To exit a streaming session, click the
              <span>
                &nbsp;
              <Button variant="contained" color="error" size="large">
                Disconnect
              </Button>
                &nbsp;
              </span>
              button.<br />
              To select a type of annotation, click the "Annotation" dropdown menu and select an annotation.
            </Typography>
            <Button
              variant="contained"
              color="secondary"
              onClick={() => setShowInstructions(false)}
            >
              Close
            </Button>
          </Box>
        )}
        </Container>
      </Box>
    </ThemeProvider>
  );
}
export default Practitioner;
