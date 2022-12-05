# Insert Acreages here
import streamlit as st
import FireSafeDb as database
import pandas as pd
import numpy as np
import pydeck as pdk
import time

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

st.header("Fire spread detector")
st.subheader("Notifies nearby residents based on the longitute and latitude of the fire and their location")


db = database.MyDatabase('Address.db')
col1, col2 = st.columns(2)
addresses = db.fetch()
longitude, latitude = addresses[0][1].split(", ")
with col1:
    st.subheader(f"Longitude: {longitude}")
with col2:
    st.subheader(f"Latitude: {latitude}")

#create numpy area with one row and two columns
acreages = np.array([[0, 0]])

df = pd.DataFrame(
    acreages/ [50,50]+ [float(longitude), float(latitude)],
    columns=['lat', 'lon'])

#create drop down
option = st.selectbox('How did it start', ['Select', 'Lightning', 'Smoking', 'Campire', 'Debris', 'Railroad', 'Powerline', 'Structure', 'Equipment'])
if option and not option == 'Select':
    time.sleep(1)
    st.write(f'You selected: {option}')
    if option == 'Lightning':
        st.write("Lightning is the most common cause of wildfires in the United States. Lightning strikes can start fires in dry grass, brush, and trees. Lightning can also start fires in homes and other structures.")
        view_state = pdk.ViewState(
            latitude=float(longitude),
            longitude=float(latitude),
            zoom=10,
            pitch=50,
        )

        #add layers to the map
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=1000*10,
        )

        #add the map to the streamlit app
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=view_state,
            layers=[layer],
        ))
    elif option == 'Campire':
        time.sleep(1)
        st.write("Campfires are the second most common cause of wildfires in the United States. Campfires can start fires in dry grass, brush, and trees. Campfires can also start fires in homes and other structures.")
        view_state = pdk.ViewState(
            latitude=float(longitude),
            longitude=float(latitude),
            zoom=10,
            pitch=50,
        )

        #add layers to the map
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=900*10,
        )

        #add the map to the streamlit app
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=view_state,
            layers=[layer],
        ))
    elif option == 'Debris':
        time.sleep(1)
        st.write("Debris burning is the third most common cause of wildfires in the United States. Debris burning is the burning of yard waste, such as leaves, grass clippings, and tree branches. Debris burning can also include the burning of agricultural waste, such as corn stalks and hay.")
        view_state = pdk.ViewState(
            latitude=float(longitude),
            longitude=float(latitude),
            zoom=10,
            pitch=50,
        )

        #add layers to the map
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=700*10,
        )

        #add the map to the streamlit app
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=view_state,
            layers=[layer],
        ))
    elif option == 'Railroad':
        time.sleep(1)
        st.write("Railroad fires are the fourth most common cause of wildfires in the United States. Railroad fires can start fires in dry grass, brush, and trees. Railroad fires can also start fires in homes and other structures.")
        view_state = pdk.ViewState(
            latitude=float(longitude),
            longitude=float(latitude),
            zoom=10,
            pitch=50,
        )

        #add layers to the map
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=500*10,
        )

        #add the map to the streamlit app
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=view_state,
            layers=[layer],
        ))
    elif option == 'Powerline':
        time.sleep(1)
        st.write("Powerline fires are the fifth most common cause of wildfires in the United States. Powerline fires can start fires in dry grass, brush, and trees. Powerline fires can also start fires in homes and other structures.")
        view_state = pdk.ViewState(
            latitude=float(longitude),
            longitude=float(latitude),
            zoom=10,
            pitch=50,
        )

        #add layers to the map
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=400*10,
        )

        #add the map to the streamlit app
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=view_state,
            layers=[layer],
        ))
    elif option == 'Structure':
        time.sleep(1)
        st.write("Structure fires are the sixth most common cause of wildfires in the United States. Structure fires can start fires in dry grass, brush, and trees. Structure fires can also start fires in homes and other structures.")
        view_state = pdk.ViewState(
            latitude=float(longitude),
            longitude=float(latitude),
            zoom=10,
            pitch=50,
        )

        #add layers to the map
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=300*10,
        )

        #add the map to the streamlit app
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=view_state,
            layers=[layer],
        ))
    elif option == 'Equipment':
        time.sleep(1)
        st.write("Equipment fires are the seventh most common cause of wildfires in the United States. Equipment fires can start fires in dry grass, brush, and trees. Equipment fires can also start fires in homes and other structures.")
        view_state = pdk.ViewState(
            latitude=float(longitude),
            longitude=float(latitude),
            zoom=10,
            pitch=50,
        )

        #add layers to the map
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=200*10,
        )

        #add the map to the streamlit app
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=view_state,
            layers=[layer],
        ))
    elif option == 'Smoking':
        time.sleep(1)
        st.write("Smoking is the eighth most common cause of wildfires in the United States. Smoking can start fires in dry grass, brush, and trees. Smoking can also start fires in homes and other structures.")
        view_state = pdk.ViewState(
            latitude=float(longitude),
            longitude=float(latitude),
            zoom=10,
            pitch=50,
        )

        #add layers to the map
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=100*10,
        )

        #add the map to the streamlit app
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=view_state,
            layers=[layer],
        ))