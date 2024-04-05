import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import MessageModal from '../modules/MessageModal';

describe('MessageModal Component', () => {
  const mockClose = jest.fn();
  const mockStopVideo = jest.fn();

  beforeEach(() => {
    render(
      <MessageModal
        isModalOpen={true}
        handleClose={mockClose}
        handelStopVideo={mockStopVideo}
      />
    );
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('renders correctly', () => {
    expect(screen.getByText(/warning/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /stop video/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument();
  });

  test('calls handleClose when the cancel button is clicked', () => {
    fireEvent.click(screen.getByRole('button', { name: /cancel/i }));
    expect(mockClose).toHaveBeenCalledTimes(1);
  });

  test('calls handelStopVideo when the stop video button is clicked', () => {
    fireEvent.click(screen.getByRole('button', { name: /stop video/i }));
    expect(mockStopVideo).toHaveBeenCalledTimes(1);
  });

  test('modal should be visible when isModalOpen is true', () => {
    expect(screen.getByText(/warning/i)).toBeVisible();
  });
});