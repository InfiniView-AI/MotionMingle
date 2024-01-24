import React, { useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Button as MUIButton } from '@mui/material';

function Button(props: { displayText: string; buttonColor: string }) {
  const { displayText, buttonColor } = props;
  return (
    <MUIButton variant="contained" color="error">
      {displayText}
    </MUIButton>
  );
}

export default Button;
