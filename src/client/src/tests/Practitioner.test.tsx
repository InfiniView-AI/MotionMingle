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

  it('renders correctly', () => {
    render(
      <Router>
        <ThemeProvider theme={theme}>
          <Practitioner />
        </ThemeProvider>
      </Router>
    );
    expect(screen.getByText('Motion Mingle')).toBeInTheDocument();
    expect(screen.getByText('Practitioner')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Connect' })).toBeInTheDocument();
  });

  it('handles annotation selection', () => {
    render(
      <Router>
        <ThemeProvider theme={theme}>
          <Practitioner />
        </ThemeProvider>
      </Router>
    );
    fireEvent.change(screen.getByTestId('select-annotation'), { target: { value: 'test' } });
    const selectElement = screen.getByTestId('select-annotation') as HTMLSelectElement;
    expect(selectElement.value).toBe('test'); 
  });

  it('initiates connection on Connect button click', async () => {
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Connect' }));
    expect(mockCreatePeerConnection).toHaveBeenCalledTimes(1);
    expect(mockConnectAsConsumer).toHaveBeenCalledTimes(0);
  });

  it('disconnects on Disconnect button click after connection', async () => {
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Connect' }));
    await user.click(screen.getByRole('button', { name: 'Disconnect' }));
    expect(screen.getByRole('button', { name: 'Connect' })).toBeInTheDocument();
  });

  it('updates video stream when new annotation is selected', async () => {
    const user = userEvent.setup();
    await user.selectOptions(screen.getByTestId('select-annotation'), ['test']);
    expect(mockConnectAsConsumer).toHaveBeenCalledWith(expect.anything(), 'test');
  });

  it('renders video element correctly', () => {
    const videoElement = screen.getByRole('video');
    expect(videoElement).toBeInTheDocument();
    expect(videoElement).toHaveAttribute('autoPlay', 'true');
  });

  it('should hide video element when not connected', () => { 
    const videoElement = screen.queryByRole('video');
    expect(videoElement).not.toBeInTheDocument();
  });

  it('should display error message when connection fails', async () => {
    mockConnectAsConsumer.mockRejectedValueOnce(new Error('Connection failed'));
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Connect' }));
    expect(screen.getByText('Connection failed')).toBeInTheDocument();
  });

});