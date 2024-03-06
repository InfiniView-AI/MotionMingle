import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import NotFound from '../modules/NotFound'; // Adjust the import path as necessary
import { BrowserRouter } from 'react-router-dom';

describe('NotFound Component', () => {
  // Render the NotFound component within a BrowserRouter to support the Link component
  beforeEach(() => {
    render(
      <BrowserRouter>
        <NotFound />
      </BrowserRouter>
    );
  });

  test('displays the not found message', () => {
    const notFoundMessage = screen.getByText(/Not Found/i);
    expect(notFoundMessage).toBeInTheDocument();
  });

  test('provides a link to the home page', () => {
    const goHomeLink = screen.getByRole('link', { name: /GO HOME/i });
    expect(goHomeLink).toBeInTheDocument();
    expect(goHomeLink).toHaveAttribute('href', '/');
  });
});