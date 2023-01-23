# FireSafe

## Always be safe, wherever you are.

<p align="center">
    <img src="https://i.imgur.com/43jd460.png"/>
</p>

[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/andyjianzhou/FireSafe?logo=github&style=for-the-badge)](https://github.com/andyjianzhou/) 
[![GitHub last commit](https://img.shields.io/github/last-commit/andyjianzhou/FireSafe?style=for-the-badge&logo=git)](https://github.com/andyjianzhou/) 
[![GitHub stars](https://img.shields.io/github/stars/andyjianzhou/FireSafe?style=for-the-badge)](https://github.com/andyjianzhou/FireSafe/stargazers) 
[![My stars](https://img.shields.io/github/stars/andyjianzhou?affiliations=OWNER%2CCOLLABORATOR&style=for-the-badge&label=My%20stars)](https://github.com/andyjianzhou/FireSafe/stargazers) 
[![GitHub forks](https://img.shields.io/github/forks/andyjianzhou/FireSafe?style=for-the-badge&logo=git)](https://github.com/andyjianzhou/FireSafe/network)
[![Code size](https://img.shields.io/github/languages/code-size/andyjianzhou/FireSafe?style=for-the-badge)](https://github.com/andyjianzhou/FireSafe)
[![Languages](https://img.shields.io/github/languages/count/andyjianzhou/FireSafe?style=for-the-badge)](https://github.com/andyjianzhou/FireSafe)
[![Top](https://img.shields.io/github/languages/top/andyjianzhou/FireSafe?style=for-the-badge&label=Top%20Languages)](https://github.com/andyjianzhou/FireSafe)
[![Issues](https://img.shields.io/github/issues/andyjianzhou/FireSafe?style=for-the-badge&label=Issues)](https://github.com/andyjianzhou/FireSafe)
[![Watchers](	https://img.shields.io/github/watchers/andyjianzhou/FireSafe?label=Watch&style=for-the-badge)](https://github.com/andyjianzhou/FireSafe/)

## FireSafe is an application designed to keep individuals in high risk areas updated on fires. It also ensures that no individual gets left behind in the event of a forest fire eruption. FireSafe uses complex machine learning to locate secluded locations and homes, allowing responders to go down a checklist of locations and ensure individuals are all aware of fire threat and prepared to evacuate.


## Features and Interfaces:
1. Landing Page
    - The landing page is the first page that the user sees when they open the application. It contains a brief description of the application and a button to get started.
    <p align="center">
    <img src="https://i.imgur.com/tB6YgJx.png"/>
    </p>
2. Login Page
    - The login page is where the user can login to their account. If they do not have an account, they can click the link to create an account.
    <!-- Insert image here later of a screenshot -->
3. Main Firesafe page:
    - The main FireSafe page has a user input textbox, a checkbox on the side, and a interactive dashboard page
    - Purposely designed for a more minimalistic, modern approach
    - Planning to add React to the project to make designs more versatile and advanced
    - ![image](https://i.imgur.com/MPSsZg5.png)

    - Then the user can input the address of the location they want to check for secluded areas near a wild fire. To make the program as dynamic as possible, we actually reetrieve the real location that most response teams want, latitude and longitude
    - ![image](https://i.imgur.com/MPSsZg5.png)

4. Machine Learning Model
    - Then user/government can drag and drop **LARGE** quantities of images, from files, or folders of images to detect for secluded areas, and for missing areas. The program will then return the images that have secluded areas, and the images that have missing areas.
    - Using an Efficientnet B0, and Efficientnet B4, two popular CNN neural network modeles to use. We can detect for secluded areas, and missing areas. The program will then return the images that have secluded areas, and the images that have missing areas.
    - Integrating MLOps was no easy task. Had to plan out for
        1. Obtaining the data
        2. Cleaning the data
        3. Training the model
        4. Inferencing theh model
        5. Deploying the model
        6. Monitoring the model
        8. Making sure model is of size and is preprocessed correctly
    - ![image](https://i.imgur.com/eFloCb9.png)

5. Evacuation Checklist
    - The evacuation checklist is a step-by-step guide for users to follow in case of a fire. It includes things like turning off gas and electricity, gathering important documents, and more. Users can also mark items as complete and receive reminders.
    - Then on the dashboard, it will create checkboxes for responders to check off, and then the program will automatically generate a checklist for responders to follow. It will send a notification to all users with the wepapp.

5. Incoming Fire Detection plans
    - FireSafe uses a complex machine learning model to predict the path of a fire based on location and how it was started. This allows users to see on a map if they are within the fire risk zone. This is in training and will be implemented in the future using Xgboosting.
    - ![image](https://i.imgur.com/y6neKIG.png)

<!-- Insert image here later of a screenshot -->
6. Notifications
- Users can sign up for notifications and receive alerts when there is a fire threat in their area. They can also choose to receive notifications via email or text message.
<!-- Insert image here later of a screenshot -->

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 


### Prerequisites

- Python 3.6 or higher
- Pytorch
- Efficientnet-b0 & Efficientnet-b4 
- Efficientnet-b0 model weights
- Nvidia Tesla P100
- Streamlit
- Typescript, React, and SQL
- OpenCV
- PILLOW


### Installing

- Clone the repository
git clone https://github.com/andyjianzhou/FireSafe

## Useful Links
<!-- Create link for demo -->
- [Demo](https://www.youtube.com/watch?v=G8i-fJkVXjM)
<!-- - [Design Document] insert FIGMA -->
