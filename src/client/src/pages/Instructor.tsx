import React, { useRef, useState } from 'react';

function Instructor() {
  const selfVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);

  const getVideo = () => {
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then((stream) => {
        // add stream to current video element
        const video = selfVideoRef.current;
        video!.srcObject = stream;
        // play video from selfVideo Ref
        video?.play();
      })
      .catch((err) => {
        console.error('OH NO!!!', err);
      });
  };

  const closeRemote = async (pc: RTCPeerConnection) => {
    pc.close();
    const tracks = await remoteVideoRef.current!.srcObject.getTracks().map((track) => track.stop());
    remoteVideoRef.current!.srcObject = null;
  };

  const closeVideo = () => {
    const video = selfVideoRef.current;
    video!.srcObject = null;
  };

  const createPeerConnection = () => {
    const config = {
      sdpSemantics: 'unified-plan',
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
    };
    const pc = new RTCPeerConnection(config);
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

  const negotiate = async (pc: RTCPeerConnection) => {
    // start
    console.log('negotiate');
    // load video
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    stream.getTracks().forEach((track) => {
      pc.addTrack(track, stream);
    });

    // negotiate
    const offer = await pc?.createOffer();
    await pc?.setLocalDescription(offer);
    const requestSdp = pc.localDescription;
    const sdp = await fetch('http://127.0.0.1:8080/offer', {
      body: JSON.stringify({
        sdp: requestSdp?.sdp,
        type: requestSdp?.type,
      }),
      headers: {
        'Content-Type': 'application/json',
      },
      method: 'POST',
    });
    // console.log(`requestSdp: ${requestSdp?.sdp}`);
    const answer = await sdp.json();
    // console.log('answer is :');
    // console.log(answer);
    await pc?.setRemoteDescription(answer);
    remoteVideoRef.current?.play();
  };

  let pc: RTCPeerConnection;

  return (
    <div className="App">
      <div className="camera">
        <video ref={selfVideoRef} width="300" height="200" playsInline>
          <track kind="captions" />
        </video>
      </div>
      <div className="result">
        <button type="button" onClick={() => getVideo()}>
          Start
        </button>
        <button type="button" onClick={() => closeVideo()}>
          Close
        </button>
        <button
          type="button"
          onClick={() => {
            pc = createPeerConnection();
            negotiate(pc);
          }}
        >
          Negotiate
        </button>
      </div>
      <div className="remote">
        <video ref={remoteVideoRef} width="300" height="200" playsInline>
          <track kind="captions" />
        </video>
      </div>
      <button type="button" onClick={() => closeRemote(pc)}>
        Close Remote
      </button>
    </div>
  );
}

export default Instructor;
