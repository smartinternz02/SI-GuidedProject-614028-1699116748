"""
A simple web app streamlit interface for the image captioning model.
run this app using the fowlloing command:
>> python -m streamlit run ./streamlit_ui.py
"""
import time
from PIL import Image

import streamlit as st
# main.py
from model_helper import ModelName, load_captioning_model, predict_caption, predict_caption_with_loop_handle

# Rest of your code goes here
# You can now use functions like load_captioning_model, predict_caption, etc.


from const import IMAGE_SIZE, MAX_LENGTH
from image import load_features_from_img

from caption import load_tokenizer

model = load_captioning_model(ModelName.EARLY_STOPPED_MODEL)
tokenizer = load_tokenizer()


st.title("Image Captioning Using CNN's + LSTM's")

st.write(
    """This is a simple web app that uses a CNN and LSTM to generate a caption for an image.The model was trained on the Flickr8k dataset.The model was trained on a Google Colab GPU.The model was trained for 30 epochs.
"""
)

# upload image button
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # placeholder container and progress bar
    container = st.empty()
    bar_ = st.progress(0)

    # reading the image and showing it
    st.write("")
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # loaind the image and getting the features
    container.text("Reading image...")
    img_features = load_features_from_img(image, IMAGE_SIZE)
    container.text("Generating caption... 0%")

    # generating the caption
    caption = "startseq"
    for i in range(MAX_LENGTH):
        percent_completed = (i + 1) / MAX_LENGTH
        container.text(f"Generating caption... {round(percent_completed, 2)}%")
        bar_.progress(i + 1)
        in_text = predict_caption_with_loop_handle(
            model,
            img_features,
            tokenizer,
            MAX_LENGTH,
            caption,
        )
        if in_text is None:
            break
        caption = in_text

    # making the progress bar full
    bar_.progress(100)
    time.sleep(0.05)

    # removing the progress bar and container
    bar_.empty()
    container.empty()

    # removing the startseq and endseq tags
    caption = caption.removeprefix("startseq").removesuffix("endseq")

    # remove trailing spaces
    caption = caption.strip()

    # captalizing the first letter of the caption
    caption = caption.capitalize()

    # showing the caption
    container.subheader(caption, divider=True)
