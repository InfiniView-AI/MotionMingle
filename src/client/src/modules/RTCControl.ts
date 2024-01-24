export const createPeerConnection = () => {
  const config = {
    sdpSemantics: 'unified-plan',
    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
  };
  const pc = new RTCPeerConnection(config);
  return pc;
};

// TODO: Need to update after we have consumer side annotation selection to server
// and need to update with data channel after we have dynamic annotation selection
export const connectAsConsumer = async (pc: RTCPeerConnection) => {
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
};
