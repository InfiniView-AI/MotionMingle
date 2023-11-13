window.onload = () => {
    document.getElementById('my-button').onclick = () => {
        init();
    }
}

async function init() {
    const peer = createPeer();
    peer.addTransceiver("video", { direction: "recvonly" })
    console.log('A practitioner joined');
}

function createPeer() {
    const peer = new RTCPeerConnection({
        iceServers: [
            {
                urls: "stun:stun.stunprotocol.org"
            }
        ]
    });
    peer.ontrack = handleTrackEvent;
    peer.onnegotiationneeded = () => handleNegotiationNeededEvent(peer);

    return peer;
}

async function handleNegotiationNeededEvent(peer) {
    const offer = await peer.createOffer();
    await peer.setLocalDescription(offer);
    const payload = {
        sdp: peer.localDescription
    };
    try {
        const { data } = await axios.post('/consumer', payload);
        const desc = new RTCSessionDescription(data.sdp);
        peer.setRemoteDescription(desc).catch(e => console.log(e));    
    } catch {
        window.alert("the instructor hasn't started streaming yet!");
    }
}

function handleTrackEvent(e) {
    document.getElementById("video").srcObject = e.streams[0];
};