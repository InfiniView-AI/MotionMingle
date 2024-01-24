import { HashRouter, Route, Routes } from 'react-router-dom';
import Home from './modules/Home';
import NotFound from './modules/NotFound';
import Instructor from './modules/Instructor';
import Practitioner from './modules/Practitioner';
import Auth from './modules/Auth';

export function App() {
  return (
    <Routes>
      <Route path="/" element={<Auth />} />
      <Route path="/home" element={<Home />} />
      <Route path="/instructor" element={<Instructor />} />
      <Route path="/practitioner" element={<Practitioner />} />
      <Route path="/auth" element={<Auth />} />
      <Route path="*" element={<NotFound />} />
      <Route path="/" element={<Auth />} />
    </Routes>
  );
}

export function WrappedApp() {
  return (
    <HashRouter>
      <App />
    </HashRouter>
  );
}
