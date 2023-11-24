# Image Captioning Model

This project utilizes `DenseNet201`, `CNN`, and `LSTM` to generate captions for images. Trained on the [Flickr8k](https://www.kaggle.com/adityajn105/flickr8k) dataset, the model undergoes 20 epochs on a GPU. Post-training, it successfully generates captions for unseen images.

## Demo
![Screenshot](assets/screenshot.png)

## Dataset
The [Flickr8k](https://www.kaggle.com/adityajn105/flickr8k) dataset, consisting of 8000 images, is employed. It's divided into 6000 training, 1000 validation, and 1000 testing images, each with five captions. The dataset is preprocessed, and captions are cleaned before model training.

## Model
The model, trained on a GPU for 20 epochs, exhibits generalization capabilities on unseen images. It comprises three main parts:

- **DenseNet201:** Used to extract features from images. The last layer is removed, and the model is pretrained on the `ImageNet` dataset.
- **Data Generation:** Generates data for training using the `DenseNet201` model to extract image features. These features are then used to train the `LSTM` model.
- **LSTM:** Generates captions for images using features extracted from images.

### DenseNet201
The `DenseNet201` model serves as a feature extractor. Pretrained on `ImageNet`, the last layer is removed, and it produces image embeddings of size 1920.

### Data Generation
Due to resource limitations, data is generated batch-wise. Image embeddings and corresponding caption text embeddings are used as inputs for training.

## LSTM
The `LSTM` model generates captions by receiving image embeddings from `DenseNet201`. Image embeddings, concatenated with the start sequence, are passed to the LSTM network, which progressively generates words to form a sentence.

## Running the Project

1. Clone the repository
   ```bash
   git clone https://github.com/Sarath191181208/image_captioning.git
