import React, { useRef, useState } from 'react';
import { Button as MUIButton } from '@mui/material';

export default function MMButton(props: {
  displayText: string;
  buttonColor: 'error' | 'primary';
  callBack: () => void;
}) {
  const { displayText, buttonColor, callBack } = props;
  return (
    <MUIButton variant="contained" color={buttonColor} onClick={callBack}>
      {displayText}
    </MUIButton>
  );
}
