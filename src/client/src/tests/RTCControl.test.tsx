import { createPeerConnection, connectAsConsumer, connectAsBroadcaster } from '../modules/RTCControl';

// Mocking the global RTCPeerConnection
global.RTCPeerConnection = jest.fn().mockImplementation(() => ({
  createOffer: jest.fn().mockResolvedValue({
    sdp: 'sdp',
    type: 'offer',
  }),
  setLocalDescription: jest.fn(),
  setRemoteDescription: jest.fn(),
  localDescription: {
    sdp: 'sdp',
    type: 'offer',
  },
  addEventListener: jest.fn(),
  addTrack: jest.fn(),
  close: jest.fn(),
}));

// Mocking the global fetch function
global.fetch = jest.fn().mockResolvedValue({
  json: jest.fn().mockResolvedValue({
    sdp: 'mocked sdp',
    type: 'answer',
  }),
});

describe('RTCControl', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('createPeerConnection', () => {
    it('should create a new RTCPeerConnection with the specified config', () => {
      const pc = createPeerConnection();
      expect(pc).toBeInstanceOf(RTCPeerConnection);
      expect(global.RTCPeerConnection).toHaveBeenCalledWith({
        sdpSemantics: 'unified-plan',
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      });
    });
  });

  describe('connectAsConsumer', () => {
    it('should create an offer, set local and remote descriptions', async () => {
      const pc = createPeerConnection();
      await connectAsConsumer(pc, 'annotation');
      expect(pc.createOffer).toHaveBeenCalled();
      expect(pc.setLocalDescription).toHaveBeenCalled();
      expect(global.fetch).toHaveBeenCalledWith(expect.any(String), expect.any(Object));
      expect(pc.setRemoteDescription).toHaveBeenCalled();
    });
  });

  describe('connectAsBroadcaster', () => {
    it('should create an offer, set local and remote descriptions for broadcasting', async () => {
      const pc = createPeerConnection();
      await connectAsBroadcaster(pc);
      expect(pc.createOffer).toHaveBeenCalled();
      expect(pc.setLocalDescription).toHaveBeenCalled();
      expect(global.fetch).toHaveBeenCalledWith(expect.any(String), expect.any(Object));
      expect(pc.setRemoteDescription).toHaveBeenCalled();
    });
  });
});