import React from 'react';
import { ThemeProvider, createTheme } from '@mui/material';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BrowserRouter as Router } from 'react-router-dom';
import Practitioner from '../modules/Practitioner';
import * as RTCControl from '../modules/RTCControl';
import { userEvent } from '@testing-library/user-event';

const theme = createTheme({
  palette: {
    primary: {
      main: '#00a0b2',
    },
    secondary: {
      main: '#E6F7FF',
    },
  },
});

jest.mock('../modules/SelectAnnotation', () => (props: { selectedAnnotation: string | number | readonly string[] | undefined; selectionHandler: React.ChangeEventHandler<HTMLSelectElement> | undefined; }) => (
  <select data-testid="select-annotation" value={props.selectedAnnotation} onChange={props.selectionHandler}>
    <option value="none">None</option>
    <option value="test">Test</option>
  </select>
));

describe('Practitioner Component', () => {
  const mockCreatePeerConnection = jest.fn();
  const mockConnectAsConsumer = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    render(
        <Router>
            <ThemeProvider theme={theme}>
                <Practitioner />
            </ThemeProvider>
        </Router>
    );
    jest.spyOn(RTCControl, 'createPeerConnection').mockImplementation(mockCreatePeerConnection);
    jest.spyOn(RTCControl, 'connectAsConsumer').mockImplementation(mockConnectAsConsumer);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should hide video element when not connected', () => { 
    const videoElement = screen.queryByRole('video');
    expect(videoElement).not.toBeInTheDocument();
  });

});