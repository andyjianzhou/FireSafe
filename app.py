import streamlit as st
import random
from streamlit.components.v1 import html
from PIL import Image
import requests
import json
import pprint

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
        background-image: url("https://cdn.dribbble.com/users/2272349/screenshots/7207200/ocean-guardian-pulse-loop-by-the-sound-of-breaking-glass.gif");
        background-size: cover;
    }
</style>
'''

st.markdown(background_img, unsafe_allow_html=True)

st.title('Hackathon App Title Here')

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
        LABELS = ['Mali', 'Ethiopia', 'Malawi', 'Nigeria']
        # display image
        if uploaded_file is not None:
            #image path
            for i, uploaded_file in enumerate(uploaded_file):
                image, width, height = load_image(uploaded_file)
                total_images.append(image)
                st.image(image, caption= i+1, use_column_width=True)
                st.write("")
                st.write("Classifying...")
                label = random.choice(LABELS)
                st.write(label)
                st.write("")

with col2:
    if found:
        col2.header(address)
        col2.subheader(f"Latitude: {str(location['results'][0]['lat'])} Longitude: {str(location['results'][0]['lon'])}")
    else:
        col2.header("No location found")
    if total_images:
        for i in range(len(total_images)):
            st.checkbox(str(i+1))
