import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import SelectAnnotation from '../modules/SelectAnnotation';

describe('SelectAnnotation Component', () => {
  const mockSelectionHandler = jest.fn();

  beforeEach(() => {
    render(<SelectAnnotation selectedAnnotation="" selectionHandler={mockSelectionHandler} />);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('renders correctly', () => {
    expect(screen.getByLabelText('Annotation')).toBeInTheDocument();
  });
});