import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Instructor from '../modules/Instructor';
import * as RTCControl from '../modules/RTCControl';
import { BrowserRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';

jest.mock('../modules/RTCControl', () => ({
  connectAsConsumer: jest.fn(),
  createPeerConnection: jest.fn().mockReturnValue({ 
    addEventListener: jest.fn(),
    addTrack: jest.fn(),
    close: jest.fn(),
  }),
  connectAsBroadcaster: jest.fn(),
}));

jest.mock('../modules/MessageModal', () => (props: { handleClose: React.MouseEventHandler<HTMLButtonElement> | undefined; handelStopVideo: React.MouseEventHandler<HTMLButtonElement> | undefined; }) => (
  <div>
    MessageModal Mock
    <button onClick={props.handleClose}>Close</button>
    <button onClick={props.handelStopVideo}>Stop Video</button>
  </div>
));

jest.mock('../modules/SelectAnnotation', () => (props: { selectionHandler: React.ChangeEventHandler<HTMLSelectElement> | undefined; }) => (
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

});