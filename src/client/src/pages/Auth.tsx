import React, { useRef, useState } from 'react';
import { Link } from 'react-router-dom';

const Auth = () => {
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
    // const tracks = await remoteVideoRef.current!.srcObject.getTracks().map((track) => track.stop());
    remoteVideoRef.current!.srcObject = null;
  };

  const closeVideo = () => {
    const video = selfVideoRef.current;
    video!.srcObject = null;
  };

  const createConsumerPeerConnection = () => {
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

  const createPeerConnection = () => {
    const config = {
      sdpSemantics: 'unified-plan',
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
    };
    const pc = new RTCPeerConnection(config);
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
    const offer = await pc?.createOffer();
    await pc?.setLocalDescription(offer);
    const requestSdp = pc.localDescription;
    const sdp = await fetch('http://127.0.0.1:8080/broadcast', {
      body: JSON.stringify({
        sdp: requestSdp?.sdp,
        type: requestSdp?.type,
        // video transform
        video_transform: 'skeleton',
      }),
      headers: {
        'Content-Type': 'application/json',
      },
      method: 'POST',
    });
    const answer = await sdp.json();
    await pc?.setRemoteDescription(answer);
    remoteVideoRef.current?.play();
  };

  const consume = async (pc: RTCPeerConnection) => {
    const offer = await pc?.createOffer();
    await pc?.setLocalDescription(offer);
    const requestSdp = pc.localDescription;
    const sdp = await fetch('http://127.0.0.1:8080/consumer', {
      body: JSON.stringify({
        sdp: requestSdp?.sdp,
        type: requestSdp?.type,
      }),
      headers: {
        'Content-Type': 'application/json',
      },
      method: 'POST',
    });
    const answer = await sdp.json();
    await pc?.setRemoteDescription(answer);
    remoteVideoRef.current?.play();
  };

  let consumer: RTCPeerConnection;

  return (
<>
    <button>
        <Link to="/instructor">Instructor</Link>
    </button>
    <button>
        <Link to="/practitioner">Practitioner</Link>
    </button>
</>
  );
}

export default Auth;
