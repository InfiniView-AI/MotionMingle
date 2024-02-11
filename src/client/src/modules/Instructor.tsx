import React, { useRef, useState, useEffect } from 'react';
import {
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  SelectChangeEvent,
  Button,
} from '@mui/material';
import MessageModal from './MessageModal';
import { connectAsConsumer, createPeerConnection, connectAsBroadcaster } from './RTCControl';

function Instructor() {
  const selfVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);

  const [selectedAnnotation, setSelectedAnnotation] = useState<string>('');
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isSelfVideoOn, setIsSelfVideoOn] = useState<boolean>(true);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [brodcastPc, setBroadcastPc] = useState<RTCPeerConnection>();

  const selectNewAnnotation = (event: SelectChangeEvent) => {
    setSelectedAnnotation(event.target.value);
  };

  const getSelfVideo = () => {
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then((stream) => {
        // add stream to current video element
        const video = selfVideoRef.current;
        video!.srcObject = stream;
        // play video from selfVideo Ref
        video?.play();
        setIsSelfVideoOn(true);
      })
      .catch((err) => {
        console.error('OH NO!!!', err);
      });
  };

  const closeSelfVideo = () => {
    const video = selfVideoRef.current;
    video!.srcObject = null;
    setIsSelfVideoOn(false);
  };

  useEffect(() => {
    getSelfVideo();
    setBroadcastPc(createPeerConnection());
  }, []);

  const closeRemote = async (pc: RTCPeerConnection) => {
    console.log(pc);
    pc.close();
    setBroadcastPc(undefined);
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

  const broadcast = async (pc: RTCPeerConnection) => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    stream.getTracks().forEach((track) => {
      pc.addTrack(track, stream);
    });
    await connectAsBroadcaster(pc);
    remoteVideoRef.current?.play();
    setIsConnected(true);
  };

  const consume = async (pc: RTCPeerConnection) => {
    await connectAsConsumer(pc, selectedAnnotation);
    remoteVideoRef.current?.play();
  };

  let consumer: RTCPeerConnection;

  return (
    <div className="App">
      Instructor
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
        </Select>
      </FormControl>
      <div className="camera">
        <video ref={selfVideoRef} width="300" height="200" playsInline>
          <track kind="captions" />
        </video>
      </div>
      {isSelfVideoOn ? (
        <Button
          variant="contained"
          color="error"
          size="large"
          onClick={() => closeSelfVideo()}
        >
          Close self video
        </Button>
      ) : (
        <Button
          variant="contained"
          color="primary"
          size="large"
          onClick={() => getSelfVideo()}
        >
          Start self video
        </Button>
      )}
      <div className="result">
        {isConnected ? (
          <Button
            variant="contained"
            color="error"
            size="large"
            onClick={() => setIsModalOpen(true)}
          >
            Stop
          </Button>
        ) : (
          <Button
            variant="contained"
            color="primary"
            size="large"
            onClick={() => {
              broadcast(brodcastPc!);
            }}
          >
            Broadcast
          </Button>
        )}
        <Button
          variant="contained"
          color="primary"
          size="large"
          onClick={() => {
            consumer = createConsumerPeerConnection();
            consume(consumer);
          }}
        >
          Check Annotated Video
        </Button>
        <Button
          variant="contained"
          color="error"
          size="large"
          onClick={() => console.log(brodcastPc)}
        >
          Mute
        </Button>
      </div>
      <div className="remote">
        <video ref={remoteVideoRef} width="300" height="200" playsInline>
          <track kind="captions" />
        </video>
      </div>
      <MessageModal
        isModalOpen={isModalOpen}
        handleClose={() => setIsModalOpen(false)}
        handelStopVideo={() => {
          closeRemote(brodcastPc!);
          setIsModalOpen(false);
        }}
      />
    </div>
  );
}

export default Instructor;
