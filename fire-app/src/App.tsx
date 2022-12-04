import React from 'react';
import logo from './logo.svg';
import './App.css';
import './components/styles/styles';
import videoBg from './assets/video.mp4';
import { CardBody, CardContainer } from './components/styles/styles';
function App() {
  return (
    <>
      <div className="App">
        <div>
          <CardContainer>
            <CardBody>
              <h1>Card Title</h1>
              <p>Card Text</p>
            </CardBody>
          </CardContainer>
        </div>
      </div>
    </>
  );
}

export default App;
