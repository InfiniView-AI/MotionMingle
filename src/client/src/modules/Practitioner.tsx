import React, { useRef, useState } from 'react';
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
  },
});

function Practitioner() {
  const remoteVideoRef = useRef<HTMLVideoElement>(null);

  const [selectedAnnotation, setSelectedAnnotation] = useState<string>('');
  const [isConnected, setIsConnected] = useState<boolean>(false);

  const selectNewAnnotation = (event: SelectChangeEvent) => {
    setSelectedAnnotation(event.target.value);
  };

  const closeRemote = async (pc: RTCPeerConnection) => {
    // pc.close();
    // const tracks = await remoteVideoRef.current!.srcObject.getTracks().map((track) => track.stop());
    remoteVideoRef.current!.srcObject = null;
    setIsConnected(false);
  };

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

  const consume = async (pc: RTCPeerConnection) => {
    await connectAsConsumer(pc, selectedAnnotation);
    remoteVideoRef.current?.play();
    setIsConnected(true);
  };

  let consumer: RTCPeerConnection;

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ bgcolor: 'background.default', minHeight: '100vh' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', p: 2 }}>
          <img src={logo} alt="Motion Mingle Logo" style={{ height: 50 }} />
          <Typography variant="h4" sx={{ ml: 2 }}>
            Motion Mingle
          </Typography>
        </Box>
        <Container maxWidth="md" sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          <Typography variant="h5" gutterBottom>
            Practitioner
          </Typography>
          <SelectAnnotation selectedAnnotation={selectedAnnotation} selectionHandler={selectNewAnnotation} />
          <video
            ref={remoteVideoRef}
            autoPlay
            style={{ width: '100%', maxWidth: '600px', aspectRatio: '600/450', border: '3px solid', borderColor: 'primary.main', borderRadius: '4px', marginTop: '20px' }}
            playsInline
          />
          {isConnected ? (
            <Button variant="contained" color="error" size="large" onClick={() => closeRemote(consumer)}>
              Disconnect
            </Button>
          ) : (
            <Button variant="contained" color="primary" size="large" onClick={() => {
              consumer = createConsumerPeerConnection();
              consume(consumer);
            }}>
              Connect
            </Button>
          )}
        </Container>
      </Box>
    </ThemeProvider>
  );
}
export default Practitioner;
