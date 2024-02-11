import React, { useEffect, useRef, useState } from 'react';
import {
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  SelectChangeEvent,
  Button,
} from '@mui/material';
import MMButton from './MMButton';
import { connectAsConsumer, createPeerConnection } from './RTCControl';

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
    // await connectAsConsumer(pc);
    const offer = await pc?.createOffer();
    await pc?.setLocalDescription(offer);
    const requestSdp = pc.localDescription;
    const sdp = await fetch('http://127.0.0.1:8080/consumer', {
      body: JSON.stringify({
        sdp: requestSdp?.sdp,
        type: requestSdp?.type,
        video_transform: selectedAnnotation,
      }),
      headers: {
        'Content-Type': 'application/json',
      },
      method: 'POST',
    });
    const answer = await sdp.json();
    await pc?.setRemoteDescription(answer);
    remoteVideoRef.current?.play();
    setIsConnected(true);
  };

  let consumer: RTCPeerConnection;

  return (
    <div className="App">
      Practitioner
      <FormControl fullWidth>
        <InputLabel id="demo-simple-select-label">Annotation</InputLabel>
        <Select
          labelId="demo-simple-select-label"
          id="demo-simple-select"
          value={selectedAnnotation}
          label="Selected Annotation"
          onChange={selectNewAnnotation}
        >
          <MenuItem value="">None</MenuItem>
          <MenuItem value="skeleton">Skeleton</MenuItem>
          <MenuItem value="edges">Edges</MenuItem>
          <MenuItem value="cartoon">Cartoon</MenuItem>
          <MenuItem value="segmentation">Segmentation</MenuItem>
        </Select>
      </FormControl>
      <div className="camera" />
      <div className="result">
        {/* <Button
          variant="contained"
          color="primary"
          onClick={() => {
            consumer = createConsumerPeerConnection();
            consume(consumer);
          }}
        >
          Connect
        </Button> */}
      </div>
      <div className="remote">
        <video
          ref={remoteVideoRef}
          width="300"
          height="200"
          color="black"
          playsInline
        >
          <track kind="captions" />
        </video>
      </div>
      {isConnected ? (
        <Button
          variant="contained"
          color="error"
          size="large"
          onClick={() => {
            closeRemote(consumer);
          }}
        >
          Disconnect
        </Button>
      ) : (
        <Button
          variant="contained"
          color="primary"
          size="large"
          onClick={() => {
            consumer = createConsumerPeerConnection();
            consume(consumer);
          }}
        >
          Connect
        </Button>
      )}
      {/* <MMButton
        displayText="Motion Mingle"
        buttonColor="error"
        callBack={() => {
          consumer = createConsumerPeerConnection();
          consume(consumer);
        }}
      /> */}
    </div>
  );
}

export default Practitioner;
