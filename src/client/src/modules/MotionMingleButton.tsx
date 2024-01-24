import React, { useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Button as MUIButton } from '@mui/material';
// import { OverridableStringUnion } from '@mui/types';
import { ButtonPropsColorOverrides } from '@mui/material/Button/Button';

export default function MotionMingleButton(props: {
  displayText: string;
  buttonColor: ButtonPropsColorOverrides;
}) {
  const { displayText, buttonColor } = props;
  return (
    <MUIButton variant="contained" color="error">
      {displayText}
    </MUIButton>
  );
}
