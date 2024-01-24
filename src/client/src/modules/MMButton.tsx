import React, { useRef, useState } from 'react';
import { Button as MUIButton } from '@mui/material';

// TODO: is this needed does it save much lines of code?
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
