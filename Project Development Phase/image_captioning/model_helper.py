# model_helper.py
from enum import Enum
from pathlib import Path

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from caption import idx_to_word

class ModelName(Enum):
    LESS_TRAINED_MODEL = "model.h5"
    EARLY_STOPPED_MODEL = "model (1).h5"

def load_captioning_model(model_name: ModelName):
    model_path = Path(f"./saves/{model_name.value}")
    model = load_model(model_path)
    return model

def predict_caption_with_loop_handle(
    model,
    img_features: np.ndarray,
    tokenizer: Tokenizer,
    max_length: int,
    mut_in_text: str,
) -> str | None:
    for _ in range(1):
        sequence = tokenizer.texts_to_sequences([mut_in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = model.predict([img_features, sequence])
        y_pred = np.argmax(y_pred)

        word = idx_to_word(int(y_pred), tokenizer)

        if word is None:
            return None

        mut_in_text += " " + word

        if word == "endseq":
            return None

    return mut_in_text

def predict_caption(
    model,
    img_features: np.ndarray,
    tokenizer: Tokenizer,
    max_length: int,
):
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = model.predict([img_features, sequence])
        y_pred = np.argmax(y_pred)

        word = idx_to_word(int(y_pred), tokenizer)

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    return in_text
