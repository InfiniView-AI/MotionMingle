import React from 'react';
import { MemoryRouter as Router } from 'react-router-dom';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom/extend-expect';
import userEvent from '@testing-library/user-event';
import Auth from '../modules/Auth';

describe('Auth Component Tests', () => {
  test('renders the Motion Mingle banner with the logo', () => {
    render(
      <Router>
        <Auth />
      </Router>
    );
    expect(screen.getByText(/Motion Mingle/i)).toBeInTheDocument();
    expect(screen.getByAltText('MotionMingle Logo')).toHaveAttribute('src', expect.stringContaining('logo.jpg'));
  });

  test('renders the slogan text correctly', () => {
    render(
      <Router>
        <Auth />
      </Router>
    );
    expect(screen.getByText(/Harmony in Motion, Unity in Practice/i)).toBeInTheDocument();
  });

  test('renders Instructor and Practitioner buttons and checks their navigation', async () => {
    const user = userEvent.setup();
    render(
      <Router>
        <Auth />
      </Router>
    );
    const instructorButton = screen.getByRole('button', { name: /Instructor/i });
    const practitionerButton = screen.getByRole('button', { name: /Practitioner/i });
    expect(instructorButton).toBeInTheDocument();
    expect(practitionerButton).toBeInTheDocument();

    // Simulate user clicking the buttons and navigating
    await user.click(instructorButton);
    expect(window.location.pathname).toBe('/');

    await user.click(practitionerButton);
    expect(window.location.pathname).toBe('/');
  });

  test('ensures the animations are present and correct', () => {
    render(
      <Router>
        <Auth />
      </Router>
    );
    const logo = screen.getByAltText('MotionMingle Logo');
    expect(logo).toHaveStyle('animation: animation-107szx5 1s ease-out;');
  });
});