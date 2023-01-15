import streamlit as st
import random
from streamlit.components.v1 import html
from PIL import Image
import requests
import json
import pprint
from models.effnetutils import get_predictions, torch_to_pil
from win10toast import ToastNotifier
import time
import FireSafeDb as database

# configuration class for model and other parameters to be used 
class CFG:
    PATH = 'models/Efnetb0BestLosses.pt'
    width = 500
    height = 500
    num_classes = 4

def set_background():

        
    # background HTML markdown code
    page_bg_img = """
    <style>
    [data-testid='stAppViewContainer"] {
    background-color: #e5
    }
    </style>
    """
    st.markdown("""
            <style>
                .css-1n76uvr {
                    width: 1053px;
                }
            </style>
            """, unsafe_allow_html=True)
    st.markdown(page_bg_img, unsafe_allow_html=True)

    background_img = '''
    <style>
        .stApp {
            background-image: url("https://i.gifer.com/LSsW.gif");
            background-size: 70%;
        }
    </style>
    '''

    st.markdown(background_img, unsafe_allow_html=True)
    st.sidebar.image('firesafelogo.png', width=300)

# API key for geoapify
class API:
    """

    Args:  API_KEY (str): API key for geoapify
    """
    def __init__(self, API_KEY):
        self.API_KEY = API_KEY

    # Processing data from geoapify API
    def autocomplete(self, parameters):
        get_URL = f"https://api.geoapify.com/v1/geocode/search?text={parameters}&format=json&apiKey={self.API_KEY}"
        response = requests.get(get_URL)
        data = response.json()
        # pprint.pprint(data)
        return data

    # convert location to array
    # replace spaces with %20 and commas with %2C%20, encryption for URL to geoapify
    def location_to_API(self, location):
        location = location.replace(',', "%2C%20")
        location = location.replace(' ', "%20")
        print(location)
        return location

# load image
def load_image(image_path):
        image = Image.open(image_path)
        width, height = image.size
        return image, width, height

# notification
def notify(title, message):
    toaster = ToastNotifier() # Create a toaster object
    toaster.show_toast(title, message, icon_path=None, duration=5, threaded=True)
    time.sleep(0.01)

# Main driver code
def run_app():
    API_KEY = 'd40f2850435a4ea88e5c1d3736182c61'
    api = API(API_KEY)

    st.title('FireSafe')

    col1, col3, col2 = st.columns(3, gap='large')
    # list to store images
    total_images = []
    # boolean to check if location is found
    found = False
    with col1:
        st.write('')

        # input react js code here in future
        location = st.text_input('Enter Location')
        if location:
            location = api.location_to_API(location)
            location = api.autocomplete(location)
            st.write("Location found!")
            address = location['results'][0]['address_line2']
            found = True

            st.write("Input image for location")
            uploaded_file = st.file_uploader("Choose a file", ['png', 'jpg'], True, label_visibility='collapsed')
            LABELS = ['Non secluded', 'Secluded']
            area_number = 1

            if uploaded_file is not None:
                
                for i, uploaded_file in enumerate(uploaded_file):
                    image, width, height = load_image(uploaded_file)

                    # get model predictions and image 
                    img, preds, probability = get_predictions(image, CFG.width, CFG.height, CFG.PATH)
                    print(type(img))
                    st.write("Classifying...")
                    if 1 in preds:
                        #find accuracy
                        accuracy = []
                        for i in range(len(preds)):
                            # if prediction is 1, append the probability to accuracy
                            if preds[i] == 1: 
                                accuracy.append(probability[i])
                        st.image(img, use_column_width=True)
                        # format f string to 3 decimal places
                        st.success(f"Secluded area detected! Detected with an accuracy of: {round(int(accuracy[0]*100), 4)}%")
                        st.success(f"Area {str(area_number)} logged")
                        area_number += 1
                        total_images.append(image)
                    else:
                        st.image(img, width=width, use_column_width=True)
                        st.success(f"{LABELS[0]} area, search for other places, accuracy of {round(int(probability[0]*100), 4)}%")
                    # get predictions
                    st.write("")

    with col2:
        if found:
            # create database object and insert data 
            db = database.MyDatabase('Address.db')

            col2.header("Checklist")
            col2.subheader(address)
            col2.subheader(f"Latitude: {str(location['results'][0]['lat'])} Longitude: {str(location['results'][0]['lon'])}")
            #get id of the first row
            id = db.fetch()[0][0]
            db.delete(id)
            db.insert(location['results'][0]['lat'], location['results'][0]['lon'])
            # print(db.fetch())
        else:
            col2.header("No location found")
        if total_images:
            for i in range(len(total_images)):
                st.checkbox(f"Area {str(i+1)}")
                notify("FireSafe", "Notifying users in these areas")
if __name__ == '__main__':
    set_background()
    run_app()