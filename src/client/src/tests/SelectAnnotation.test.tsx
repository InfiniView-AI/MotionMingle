import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import SelectAnnotation from '../modules/SelectAnnotation'; // Adjust the import path as necessary

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
    expect(screen.getByRole('button', { name: 'None' })).toBeInTheDocument();
  });

  it('should have correct options', () => {
    fireEvent.mouseDown(screen.getByRole('button', { name: 'None' }));
    const options = screen.getAllByRole('option');
    expect(options.length).toBe(4); // includes the 'None' option
    expect(options[1]).toHaveTextContent('Skeleton');
    expect(options[2]).toHaveTextContent('Edges');
    expect(options[3]).toHaveTextContent('Cartoon');
  });

  it('calls selectionHandler when an option is selected', () => {
    fireEvent.mouseDown(screen.getByRole('button'));
    fireEvent.click(screen.getByText('Skeleton'));
    expect(mockSelectionHandler).toHaveBeenCalled();
    expect(mockSelectionHandler).toHaveBeenCalledWith(expect.objectContaining({ target: { value: 'skeleton' }}));
  });

  it('displays the selected annotation', () => {
    const newProps = { selectedAnnotation: 'edges', selectionHandler: mockSelectionHandler };
    render(<SelectAnnotation {...newProps} />);
    expect(screen.getByRole('button', { name: 'Edges' })).toBeInTheDocument();
  });
});