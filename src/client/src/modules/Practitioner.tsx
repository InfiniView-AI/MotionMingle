import React, { useRef, useState } from 'react';
import {
  SelectChangeEvent,
  Button,
} from '@mui/material';
import { connectAsConsumer, createPeerConnection } from './RTCControl';
import SelectAnnotation from './SelectAnnotation';

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
    <div className="App">
      Practitioner
      <SelectAnnotation selectedAnnotation={selectedAnnotation} selectionHandler={selectNewAnnotation} />
      <div className="camera" />
      <div className="result">
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
    </div>
  );
}

export default Practitioner;
