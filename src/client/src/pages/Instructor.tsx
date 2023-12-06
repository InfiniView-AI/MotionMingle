import React, { useRef, useEffect, useState } from 'react';

function Instructor() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const photoRef = useRef<HTMLCanvasElement>(null);

  const getVideo = () => {
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then((stream) => {
        const video = videoRef.current;
        video!.srcObject = stream;
        video?.play();
      })
      .catch((err) => {
        console.error('OH NO!!!', err);
      });
  };

  const closeVideo = () => {
    const video = videoRef.current;
    video!.srcObject = null;
  };

  return (
    <div className="App">
      <div className="camera">
        <video ref={videoRef}>
          <track kind="captions" />
        </video>
      </div>
      <div className="result">
        <canvas ref={photoRef} />
        <button type="button" onClick={() => getVideo()}>
          Start
        </button>
        <button type="button" onClick={() => closeVideo()}>
          Close
        </button>
      </div>
    </div>
  );
}

export default Instructor;
