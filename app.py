import streamlit as st
import random
from streamlit.components.v1 import html

page_bg_img = """
<style>
[data-testid='stAppViewContainer"] {
background-color: #e5
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.title('Hackathon App Title Here')


# Streamlit run app.py


col1, col2 = st.columns(2, gap='large')

with col1:
    st.write('')

    st.subheader('Edit Text')
    uploaded_file = st.file_uploader("Choose a file", ['png', 'img'], True, label_visibility='collapsed')
    # LABEL


with col2:
    st.write('')

    col2.subheader('Locations')
    show_checkbox = False
    st.checkbox('Place 1', value=False, key='joe 1')

    button1 = st.checkbox('Active Location 1', value=False, key='Check1', disabled=False)
    button2 = st.checkbox('Active Location 2', value=False, key='Check2', disabled=False)
    button3 = st.checkbox('Active Location 3', value=False, key='Check3', disabled=False)
    button4 = st.checkbox('Active Location 4', value=False, key='Check4', disabled=False)
    button5 = st.checkbox('Active Location 5', value=False, key='Check5', disabled=False)
    button6 = st.checkbox('Active Location 6', value=False, key='Check6', disabled=False)
    button7 = st.checkbox('Active Location 7', value=False, key='Check7', disabled=False)
    button8 = st.checkbox('Active Location 8', value=False, key='Check8', disabled=False)
