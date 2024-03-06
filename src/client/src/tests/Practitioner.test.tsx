import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BrowserRouter as Router } from 'react-router-dom';
import Practitioner from '../modules/Practitioner';
import * as RTCControl from '../modules/RTCControl';

jest.mock('../modules/SelectAnnotation', () => (props) => (
  <select data-testid="select-annotation" value={props.selectedAnnotation} onChange={props.selectionHandler}>
    <option value="none">None</option>
    <option value="test">Test</option>
  </select>
));

describe('Practitioner Component', () => {
  const mockCreatePeerConnection = jest.fn();
  const mockConnectAsConsumer = jest.fn();

  beforeEach(() => {
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
    expect(screen.getByTestId('select-annotation').value).toBe('test');
  });

});