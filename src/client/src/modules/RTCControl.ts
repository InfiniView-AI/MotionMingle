export const createPeerConnection = () => {
  const config = {
    sdpSemantics: 'unified-plan',
    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
  };
  const pc = new RTCPeerConnection(config);
  return pc;
};

export const connectAsConsumer = async (pc: RTCPeerConnection, selectedAnnotation: string) => {
  const offer = await pc?.createOffer();
  await pc?.setLocalDescription(offer);
  const requestSdp = pc.localDescription;
  const sdp = await fetch('http://192.168.40.85:5173/consume', {
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
};

export const connectAsBroadcaster = async (pc: RTCPeerConnection) => {
  const offer = await pc?.createOffer();
  await pc?.setLocalDescription(offer);
  const requestSdp = pc.localDescription;
  const sdp = await fetch('http://192.168.40.85:5173/broadcast', {
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
};
