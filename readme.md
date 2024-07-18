# Character Recognition with Convolutional Neural Networks

This project implements character recognition using Convolutional Neural Networks (CNNs) in Python. The dataset consists of character images, and the models are trained to classify these characters. The project uses the Keras library with TensorFlow backend for building and training the models.

## Dataset

The dataset is divided into training and validation sets with the following structure:

- `./dataset/characters/Train/`: Contains training images for each character.
- `./dataset/characters/Validation/`: Contains validation images for each character.

Certain characters (numbers and special characters) are excluded from the model training.

## Project Structure

- `main.py`: The main script containing the code to load the data, preprocess it, build and train the CNN models, and visualize the results.
- `dataset/characters/Train/`: Directory for training images.
- `dataset/characters/Validation/`: Directory for validation images.

## Dependencies

- numpy
- pandas
- os
- random
- time
- cv2 (OpenCV)
- imutils
- matplotlib
- seaborn
- scikit-learn
- keras
- tensorflow

You can install the dependencies using pip:

```sh
pip install numpy pandas opencv-python imutils matplotlib seaborn scikit-learn keras tensorflow
```

## Model Architecture

Three CNN models are built and compared:

1.  **Model 1**:

    - Convolutional layers: 3 layers with 32, 64, and 128 filters respectively.
    - Activation: ReLU
    - Pooling: MaxPooling2D
    - Dense layers: 128 units
    - Dropout: 0.25 and 0.2

2.  **Model 2**:

    - Convolutional layers: 3 layers with 32, 64, and 96 filters respectively.
    - Activation: Leaky ReLU
    - Pooling: MaxPooling2D
    - Dense layers: 64 units
    - Dropout: 0.25 and 0.2

3.  **Model 3**:

    - Convolutional layers: 3 layers with 32, 64, and 128 filters respectively.
    - Activation: Leaky ReLU
    - Pooling: MaxPooling2D
    - Dense layers: 128 units
    - Dropout: 0.25 and 0.2

## Training

The models are trained for 50 epochs with a batch size of 32. The training and validation accuracy and loss are plotted for each model to compare their performance.

## Results

Confusion matrices are generated to evaluate the performance of the models on the validation set.

## Usage

To run the script and train the models, execute the following command:

```
python main.py
```

## Visualizing Results

The script generates plots for training vs validation accuracy and loss for each model. Confusion matrices are also displayed for each model's predictions on the validation set.

## Character Prediction

The script includes functions to predict characters from images of words. The `get_letters` function extracts characters from an image and predicts them using the trained model.

```
letters, image = get_letters("path/to/image.jpg")
word = get_word(letters)
print(word)
plt.imshow(image)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the contributors and the open-source community for their valuable resources and support.
