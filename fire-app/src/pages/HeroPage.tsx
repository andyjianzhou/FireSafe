import React from 'react';
import logo from './logo.svg';
import '../App.css';
import '../components/styles/styles';
import '../index.d.ts';
import LoginPage from './LoginPage';

import { CardBody, CardContainer } from '../components/styles/styles';
import { BrowserRouter as Router, Route } from 'react-router-dom';
const videoBg = require('../assets/videoBg.mp4');
function App() {
  return (
    <>
    <div className="App">
        <div className="overlay"></div>
            <video src={videoBg} autoPlay loop muted/>
            <div className="content">
                <h1>
                    FireSafe
                </h1>
                <p>
                Always be safe, wherever you are, whatever you do.
                </p>
            </div>
        </div>
    </>
  );
}

export default App;
