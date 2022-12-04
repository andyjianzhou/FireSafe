import React, { useState } from 'react';
// import card
import { CardBody, CardContainer } from '../components/styles/styles';
import "./LoginElements.css"

const LoginPage = () => {
    const [popupStyle, showPopup] = useState("hide")

    const popup = () => {
        showPopup("login-popup")
        setTimeout(() => {
            showPopup("hide")
        }, 3000)
        // redirect to external website
        window.location.href = "http://localhost:8501"
        
    }
    return (
        <>
            <CardContainer>
                <CardBody>
                    <div className="login-cover">
                        <h1>Login</h1>
                        <input type="text" placeholder="Name..." />
                        <input type="text" placeholder="Address...Optional" />
                        <input type="password" placeholder="Password..." />
                        <input type="tel" placeholder="Phone..." />
                        
                        <div className="login-btn" onClick = {popup}>Login</div>
                        
                        <p className="text">Or login using</p>

                        <div className="alt-login">
                            <div className="facebook"></div>
                            <div className="google"></div>
                        </div>

                        <div className={popupStyle}>
                            <h3>Login Sucessful!</h3>
                            <p>Welcome Andy Zhou!</p> 
                        </div>
                    </div>
                </CardBody>
            </CardContainer>
        </>
    )
}

export default LoginPage;