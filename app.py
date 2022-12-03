import streamlit as st
import random
from streamlit.components.v1 import html
from PIL import Image
import requests
import json

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


def autocomplete(parameters, API_KEY):
    get_URL = f"https://maps.googleapis.com/maps/api/place/autocomplete/output?{parameters}&key={API}"


col1, col3, col2 = st.columns(3, gap='large')
total_images = []
def load_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return image, width, height

with col1:
    st.write('')

    location = st.text_input('Enter your location', value='')

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
    # button1 = st.checkbox('Active Location 1', value=False, key='Check1', disabled=False)
    # button2 = st.checkbox('Active Location 2', value=False, key='Check2', disabled=False)
    # button3 = st.checkbox('Active Location 3', value=False, key='Check3', disabled=False)
    # button4 = st.checkbox('Active Location 4', value=False, key='Check4', disabled=False)
    # button5 = st.checkbox('Active Location 5', value=False, key='Check5', disabled=False)
    # button6 = st.checkbox('Active Location 6', value=False, key='Check6', disabled=False)
    # button7 = st.checkbox('Active Location 7', value=False, key='Check7', disabled=False)
    # button8 = st.checkbox('Active Location 8', value=False, key='Check8', disabled=False)
