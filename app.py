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

st.title('Hackathon App Title Here')

API_KEY = 'd40f2850435a4ea88e5c1d3736182c61'
def autocomplete(parameters, API_KEY):
    get_URL = f"https://api.geoapify.com/v1/geocode/search?name={parameters}apiKey={API_KEY}"
    response = requests.get(get_URL)
    data = response.json()
    pprint.pprint(data)

autocomplete('Gaskell Rd, Rosamond', API_KEY)
col1, col3, col2 = st.columns(3, gap='large')


total_images = []
def load_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return image, width, height

with col1:
    st.write('')

    # input react js code here
    location = st.text_input('Enter Location')
    if location.casefold() == 'Gaskell Rd, Rosamond'.casefold():
        st.write("Location found!")
        st.write("Location is" + "21505 Gaskell Rd, Rosamond, CA 93560, United States")
    if location:
        st.write("Input image for location")
        uploaded_file = st.file_uploader("Choose a file", ['png', 'jpg'], True, label_visibility='collapsed')
        LABELS = ['Mali', 'Ethiopia', 'Malawi', 'Nigeria']
        # display image
        if uploaded_file is not None:
            #image path
            for uploaded_file in uploaded_file:
                image, width, height = load_image(uploaded_file)
                total_images.append(image)
                st.image(image, caption='Uploaded Image.', use_column_width=True)
                st.write("")
                st.write("Classifying...")
                label = random.choice(LABELS)
                st.write(label)
                st.write("")

with col2:
    st.write('')

    col2.subheader('Locations')
    if total_images:
        for i in range(len(total_images)):
            st.checkbox(str(i))
