import React, { useRef, useState, useEffect } from 'react';
import { SelectChangeEvent,
  Button,
  Typography,
  Box,
  Container,
  ThemeProvider,
  createTheme } from '@mui/material';
import MessageModal from './MessageModal';
import SelectAnnotation from './SelectAnnotation';
import { connectAsConsumer, createPeerConnection, connectAsBroadcaster } from './RTCControl';
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


function Instructor() {
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

  const selfVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);

  const [selectedAnnotation, setSelectedAnnotation] = useState<string>('');
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isPlayingRemoteVideo, setIsPlayingRemoteVideo] = useState<boolean>(false);
  const [isSelfVideoOn, setIsSelfVideoOn] = useState<boolean>(true);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [broadcasterPc, setBroadcasterPc] = useState<RTCPeerConnection>(createPeerConnection());
  const [consumerPc, setConsumerPc] = useState<RTCPeerConnection>(createConsumerPeerConnection());
  const [showInstructions, setShowInstructions] = useState(false);

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
  }, []);

  const stopBroadcasting = async () => {
    broadcasterPc.close();
    // Replace the original peerconnection to a new peerconnection candidate
    setBroadcasterPc(createPeerConnection());
    setIsConnected(false);
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

  const broadcastLocalVideo = async () => {
    const videoElement = document.createElement('video');

    videoElement.src = tai_chi_video;

    videoElement.onloadedmetadata = async () => {

      const stream = (videoElement as any).captureStream();
      // broadcasterPc.getSenders().forEach(sender => {
      //   broadcasterPc.removeTrack(sender);
      // });

      stream.getTracks().forEach((track: MediaStreamTrack) => {
        broadcasterPc.addTrack(track, stream);
      });
      await connectAsBroadcaster(broadcasterPc);
      if (remoteVideoRef.current) remoteVideoRef.current.srcObject = stream;
      setIsConnected(true);
    };

    videoElement.load();
  };

  const consume = async () => {
    await connectAsConsumer(consumerPc, selectedAnnotation);
    remoteVideoRef.current?.play();
    setIsPlayingRemoteVideo(true);
  };

  const closeRemoteVideo = () => {
    remoteVideoRef.current!.srcObject = null;
    setIsPlayingRemoteVideo(false);
  }

  useEffect(() => {
    if(isConnected) {
      // closeRemote();
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
            Instructor
          </Typography>
        <SelectAnnotation selectedAnnotation={selectedAnnotation} selectionHandler={selectNewAnnotation} />
        <video
          ref={selfVideoRef}
          autoPlay
          style={{ width: '100%', maxWidth: '800px', aspectRatio: '600/450', border: '3px solid', borderColor: 'primary.main', borderRadius: '4px', marginTop: '20px' }}
          playsInline
        />
        {isPlayingRemoteVideo ? (
          <div className="remote">
        <video ref={remoteVideoRef} width="300" height="200" playsInline>
          <track kind="captions" />
        </video>
        </div>
        ) : null}

                {/* <video src="11_forms_demo_4min.mp4" controls playsInline/> */}
        {
          /*
          <video width="750" height="500" controls >
            <source src={video} type="video/mp4"/>
          </video>
          */
        }
        
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
              broadcast(broadcasterPc!);
            }}
          >
            Broadcast
          </Button>
        )}

        {isPlayingRemoteVideo ? (
        <Button
          variant="contained"
          color="error"
          size="large"
          onClick={() => closeRemoteVideo()}
        >
          Close Annotated Video
        </Button>
        
        ): (
        <Button
          variant="contained"
          color="primary"
          size="large"
          onClick={() => {
            consume();
          }}
          disabled={!isConnected}
        >
          Check Annotated Video
        </Button>
        )}
        {/*
        <Button
          variant="contained"
          color="secondary"
          size="large"
          onClick={broadcastLocalVideo}
        >
          Broadcast Local Video
        </Button>
         */}
      </div>
        <MessageModal
          isModalOpen={isModalOpen}
          handleClose={() => setIsModalOpen(false)}
          handelStopVideo={() => {
            stopBroadcasting();
            setIsModalOpen(false);
          }}
        />
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
            <Typography variant="body1" style={{ fontSize: '25px', lineHeight: '2' }} gutterBottom >
            To start a streaming session, click the
              <span>
                &nbsp;
                <Button variant="contained" color="primary" size="large">
                  Broadcast
                </Button>
                &nbsp;
              </span> 
              button.<br />
              To end an ongoing streaming session, click the 
              <span>
                &nbsp;
                <Button variant="contained" color="error" size="large">
                  Stop
                </Button>
                &nbsp;
              </span>
               button.<br />
              To turn on/off the self video, click 
              <span>
                &nbsp;
                <Button variant="contained" color="primary" size="large">
                  Start self video
                </Button>
                &nbsp;
              </span>
              or
              <span>
                &nbsp;
                <Button variant="contained" color="error" size="large">
                  Close self video
                </Button>
                &nbsp;
              </span>
              button.<br />
              To check annotated video, ensure you are broadcasting, select your desired type of annotation from the "Annotation" dropdown menu, then click the
              <span>
                &nbsp;
                <Button variant="contained" color="primary" size="large">
                  Check Annotated Video
                </Button>
                &nbsp;
              </span>
               button.<br />
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

export default Instructor;
