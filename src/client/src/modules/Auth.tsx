import React, { useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Button } from '@mui/material';

function Auth() {
  return (
    <center>
      <h1>Hi there,</h1>
      <h1>Welcome to MotionMingle!</h1>
      <h1>Are you a Tai Chi Instructor or a Practitioner?</h1>
      <h1>I am </h1>
      <Button variant="contained" color="primary" style={{ margin: 30 }}>
        <Link
          to="/instructor"
          style={{ color: '#FFF', textDecoration: 'none' }}
        >
          Instructor
        </Link>
      </Button>
      <span>or</span>
      <Button variant="contained" color="primary" style={{ margin: 30 }}>
        <Link
          to="/practitioner"
          style={{ color: '#FFF', textDecoration: 'none' }}
        >
          Practitioner
        </Link>
      </Button>
    </center>
  );
}

export default Auth;
