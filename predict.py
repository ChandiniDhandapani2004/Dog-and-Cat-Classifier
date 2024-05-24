import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model('dog_classifier_model.h5')

# Function to predict class of an image
def predict_image_class(image_path):
    # Load image from local file path
    img = Image.open(image_path)
    
    # Resize the image to 150x150
    img = img.resize((150, 150))
    
    # Convert image to array and preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # Predict the class
    prediction = model.predict(img_array)
    
    # Display the prediction
    if prediction < 0.5:
        print("It's a cat!")
    elif prediction > 0.5:
        print("It's a dog!")
    else:
        print("Model not trained for this instance.")

# Example usage
image_path = 'train\cat\cat.10259.jpg'
predict_image_class(image_path)
