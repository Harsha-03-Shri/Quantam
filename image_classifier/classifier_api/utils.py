# image_classifier/classifier_api/utils.py
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # <-- added
from PIL import Image
import io
import base64

def load_model():
    """Load the Keras model."""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model.keras')
    
    custom_objects = {
        'LeakyReLU': keras.layers.LeakyReLU,
        'leaky_relu': keras.activations.relu
    }
    
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded successfully. Input shape: {model.input_shape}")
        return model
    except Exception as e:
        print(f"Error loading model with standard method: {e}")
        try:
            model = keras.Sequential([
                keras.layers.InputLayer(input_shape=(28, 28, 1)),
                keras.layers.Conv2D(32, (3, 3), activation=keras.layers.LeakyReLU(alpha=0.01)),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.5),
                keras.layers.Conv2D(64, (3, 3), activation=keras.layers.LeakyReLU(alpha=0.01)),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.4),
                keras.layers.Conv2D(128, (3, 3), activation=keras.layers.LeakyReLU(alpha=0.01)),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation=keras.layers.LeakyReLU(alpha=0.001)),
                keras.layers.Dense(64, activation=keras.layers.LeakyReLU(alpha=0.001)),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(10, activation='softmax')
            ])
            try:
                model.load_weights(model_path)
                print("Model recreated and weights loaded successfully")
            except:
                print("Warning: Could not load weights, using uninitialized model")
                
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e2:
            print(f"Error in fallback model creation: {e2}")
            print("WARNING: Using emergency fallback model with random initialization")
            return create_emergency_model()

def create_emergency_model():
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(image_data):
    """
    Preprocess base64 or byte stream image for the model.
    """
    try:
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
            image_data = base64.b64decode(image_data)
        
        img = Image.open(io.BytesIO(image_data))
        print(f"Original image: {img.size}, {img.mode}")
        img = img.resize((28, 28))

        if img.mode != 'L':
            img = img.convert('L')

        img_array = np.array(img)
        print(f"Array shape: {img_array.shape}, min: {img_array.min()}, max: {img_array.max()}")
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)
        print(f"Final array shape: {img_array.shape}")

        return img_array
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return np.zeros((1, 28, 28, 1))

def preprocess_local_image(path):
    """
    Preprocess a local image file for model inference.
    """
    try:
        img = load_img(path, color_mode='grayscale', target_size=(28, 28))
        img = img_to_array(img)
        img = img.reshape(1, 28, 28, 1)
        img = img.astype('float32') / 255.0
        return img
    except Exception as e:
        print(f"Error in preprocessing local image: {e}")
        return np.zeros((1, 28, 28, 1))

def predict_class(image_data):
    """
    Predict class from base64 or byte stream image.
    """
    try:
        model = load_model()
        processed_image = preprocess_image(image_data)
        prediction = model.predict(processed_image)

        if prediction.shape[-1] > 2:
            class_index = np.argmax(prediction[0])
            confidence = float(prediction[0][class_index])
            classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        else:
            class_index = int(prediction[0][0] > 0.5)
            confidence = float(prediction[0][0] if class_index else 1 - prediction[0][0])
            classes = ['sandal', 'ankle boot']
        
        return {
            'class': classes[class_index],
            'confidence': confidence
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {
            'class': 'Error: Could not classify image',
            'confidence': 0.0,
            'error': str(e)
        }
