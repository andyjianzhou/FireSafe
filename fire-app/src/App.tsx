import React from 'react';
import logo from './logo.svg';
import './App.css';
import './components/styles/styles';
import './index.d.ts';
import Home from './pages/Home';
// import the videoBg.mp4 file
// iclude index.d.ts file

import { CardBody, CardContainer } from './components/styles/styles';
//import routers
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
// import Login from './pages/Login';
import HeroPage from './pages/HeroPage';
import LoginPage from './pages/LoginPage';

const videoBg = require('./assets/videoBg.mp4');
function App() {
  return (
    <>
    {/* Route pages but also initialize them */}
    <Router>
      <Home/>

    </Router>

    
    </>
  );
}

export default App;
