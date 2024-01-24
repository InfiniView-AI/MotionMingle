import React, { useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Button } from '@mui/material';

function Auth() {
  return (
    <>
      <h1>Hi there,</h1>
      <h1>Welcome to MotionMingle!</h1>
      <h1>Are you a Tai Chi instructor or a practitioner?</h1>
      <h1>Hi! I'm a </h1>
      <Button variant="contained" color="primary">
        <Link
          to="/instructor"
          style={{ color: '#FFF', textDecoration: 'none' }}
        >
          Instructor
        </Link>
      </Button>
      <Button variant="contained" color="primary">
        <Link
          to="/practitioner"
          style={{ color: '#FFF', textDecoration: 'none' }}
        >
          Practitioner
        </Link>
      </Button>
    </>
  );
}

export default Auth;
