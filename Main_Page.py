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
def notify(title, message):
    toaster = ToastNotifier()
    toaster.show_toast(title, message, icon_path=None, duration=5, threaded=True)
    time.sleep(0.01)

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
class CFG:
    PATH = 'models/Efnetb0BestLosses.pt'
    width = 500
    height = 500
    num_classes = 4

st.title('FireSafe')

API_KEY = 'd40f2850435a4ea88e5c1d3736182c61'
def autocomplete(parameters, API_KEY):
    get_URL = f"https://api.geoapify.com/v1/geocode/search?text={parameters}&format=json&apiKey={API_KEY}"
    response = requests.get(get_URL)
    data = response.json()
    pprint.pprint(data)
    return data

# convert location to array
def location_to_array(location):
    location = location.replace(',', "%2C%20")
    location = location.replace(' ', "%20")
    print(location)
    return location

col1, col3, col2 = st.columns(3, gap='large')
total_images = []
def load_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return image, width, height

found = False
with col1:
    st.write('')

    # input react js code here
    location = st.text_input('Enter Location')
    if location:
        location = location_to_array(location)
        location = autocomplete(location, API_KEY)
        st.write("Location found!")
        address = location['results'][0]['address_line2']
        found = True

    if location:
        st.write("Input image for location")
        uploaded_file = st.file_uploader("Choose a file", ['png', 'jpg'], True, label_visibility='collapsed')
        LABELS = ['Non secluded', 'Secluded']
        # display image
        if uploaded_file is not None:
            #image path
            for i, uploaded_file in enumerate(uploaded_file):
                image, width, height = load_image(uploaded_file)
                img, preds, probability = get_predictions(image, CFG.width, CFG.height, CFG.PATH)
                print(type(img))
                st.write("Classifying...")
                # convert tensor image to PIL image
                if 1 in preds:
                    #find accuracy
                    accuracy = []
                    for i in range(len(preds)):
                        if preds[i] == 1:
                            accuracy.append(probability[i])
                    st.image(img, use_column_width=True)
                    # format f string to 3 decimal places
                    st.success(f"Secluded area detected! Detected with an accuracy of: {round(int(probability[0]*100), 4)}%")
                    total_images.append(image)
                else:
                    st.image(img, use_column_width=True)
                    st.success(f"{LABELS[0]} area, search for other places, accuracy of {round(int(probability[0]*100), 4)}%")
                # get predictions
                st.write("")

with col2:
    if found:
        col2.header(address)
        col2.subheader(f"Latitude: {str(location['results'][0]['lat'])} Longitude: {str(location['results'][0]['lon'])}")
    else:
        col2.header("No location found")
    if total_images:
        for i in range(len(total_images)):
            st.checkbox(f"Area {str(i+1)}")
            notify("FireSafe", "Secluded area detected! Users near this location has been detected")
