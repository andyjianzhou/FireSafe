import streamlit as st
import random
st.title('Hackathon App Title Here')


# Streamlit run app.py


col1, col2 = st.columns(2)
with col1:
    st.header('Image Drop')
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        st.write(bytes_data)

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        st.write(string_data)

        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)


with col2:
    col2.header('Locations')
    show_checkbox = False
    st.checkbox('Place 1', value=False, key='joe 1')

    # if key:
    #
    #     button1 = st.checkbox('Box 1', value=False, key='True')
    #     button2 = st.checkbox('Box 2', value=False, key='True')
    #     button3 = st.checkbox('Box 3', value=False, key='True')
    #     button4 = st.checkbox('Box 4', value=False, key='True')
    #     button5 = st.checkbox('Box 5', value=False, key='True')
    #     button6 = st.checkbox('Box 6', value=False, key='True')
    #     button7 = st.checkbox('Box 7', value=False, key='True')





