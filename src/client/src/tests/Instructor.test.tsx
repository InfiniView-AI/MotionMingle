import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Instructor from '../modules/Instructor'; // Adjust the import path as necessary
import { BrowserRouter } from 'react-router-dom';

// Mock the modules used in Instructor component
jest.mock('../modules/RTCControl', () => ({
  connectAsConsumer: jest.fn(),
  createPeerConnection: jest.fn().mockReturnValue({ 
    addEventListener: jest.fn(),
    addTrack: jest.fn(),
    close: jest.fn(),
  }),
  connectAsBroadcaster: jest.fn(),
}));

jest.mock('../modules/MessageModal', () => (props) => (
  <div>
    MessageModal Mock
    <button onClick={props.handleClose}>Close</button>
    <button onClick={props.handelStopVideo}>Stop Video</button>
  </div>
));

jest.mock('../modules/SelectAnnotation', () => (props) => (
  <select data-testid="select-annotation" onChange={props.selectionHandler}>
    <option value="">Select an option</option>
    <option value="annotation1">Annotation 1</option>
    <option value="annotation2">Annotation 2</option>
  </select>
));

describe('Instructor Component', () => {
  beforeEach(() => {
    render(
      <BrowserRouter>
        <Instructor />
      </BrowserRouter>
    );
  });

  test('renders without crashing', () => {
    expect(screen.getByText('Motion Mingle')).toBeInTheDocument();
  });

  test('can select an annotation', async () => {
    fireEvent.change(screen.getByTestId('select-annotation'), { target: { value: 'annotation1' } });
    await waitFor(() => {
      expect(screen.getByTestId('select-annotation').value).toBe('annotation1');
    });
  });

  test('shows MessageModal when clicking Stop while connected', async () => {
    fireEvent.click(screen.getByText('Broadcast'));
    fireEvent.click(screen.getByText('Stop'));
    expect(screen.getByText('MessageModal Mock')).toBeInTheDocument();
  });

  test('self video starts and stops correctly', async () => {
    const startButton = screen.getByText('Start self video');
    fireEvent.click(startButton);
    const closeButton = await screen.findByText('Close self video');
    expect(closeButton).toBeInTheDocument();
    fireEvent.click(closeButton);
    expect(startButton).toBeInTheDocument();
  });

});