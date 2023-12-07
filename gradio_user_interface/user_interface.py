import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
import gradio as gr


classes_dict = {0: 'Apple___Apple_scab',
 1: 'Apple___Black_Rot',
 2: 'Apple___Cedar_Apple_rust',
 3: 'Apple___Healthy',
 4: 'Blueberry___Healthy',
 5: 'Cherry_(including_sour)___Powdery_Mildew',
 6: 'Cherry_(including_sour)___Healthy',
 7: 'Corn_(maize)___Cercospora_Leaf_Spot Gray_Leaf_Spot',
 8: 'Corn_(maize)___Common_Rust_',
 9: 'Corn_(maize)___Northern_Leaf_Blight',
 10: 'Corn_(maize)___Healthy',
 11: 'Grape___Black_Rot',
 12: 'Grape___Esca_(Black_Measles)',
 13: 'Grape___Leaf_Llight_(Isariopsis_Leaf_Spot)',
 14: 'Grape___Healthy',
 15: 'Orange___Haunglongbing_(Citrus_Greening)',
 16: 'Peach___Bacterial_Spot',
 17: 'Peach___Healthy',
 18: 'Pepper,_Bell___Bacterial_Spot',
 19: 'Pepper,_Bell___Healthy',
 20: 'Potato___Early_Blight',
 21: 'Potato___Late_Blight',
 22: 'Potato___Healthy',
 23: 'Raspberry___Healthy',
 24: 'Soybean___Healthy',
 25: 'Squash___Powdery_Mildew',
 26: 'Strawberry___Leaf_Scorch',
 27: 'Strawberry___Healthy',
 28: 'Tomato___Bacterial_Spot',
 29: 'Tomato___Early_Blight',
 30: 'Tomato___Late_Blight',
 31: 'Tomato___Leaf_Mold',
 32: 'Tomato___Septoria_Leaf_Spot',
 33: 'Tomato___Spider_Mites Two-Spotted_Spider_Mite',
 34: 'Tomato___Target_Spot',
 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 36: 'Tomato___Tomato_Mosaic_Virus',
 37: 'Tomato___Healthy'}


# Remove underscores from the dictionary values
classes_dict_cleaned = {key: value.replace('_', ' ') for key, value in classes_dict.items()}

# Now, classes_dict_cleaned contains the cleaned class names
# print(classes_dict_cleaned)


# Function to make predictions

def prediction(path, model, classes_dict):
    # Load and preprocess the image
    img = load_img(path, target_size=(256, 256))
    img_arr = img_to_array(img)
    processed_img_arr = preprocess_input(img_arr)

    # Expand image dimensions
    img_exp_dim = np.expand_dims(processed_img_arr, axis=0)

    # Make predictions using the model
    pred = np.argmax(model.predict(img_exp_dim))

    # Get the predicted class label
    predicted_class = classes_dict[pred]

    # Make predictions using the model
    prediction_probabilities = model.predict(img_exp_dim)
    pred_class_index = np.argmax(prediction_probabilities)
    predicted_probability = prediction_probabilities[0][pred_class_index]

    # Plot the input image
    plt.imshow(img)
    plt.axis('off')  # Remove axis
    plt.title(f"Predicted Class: {predicted_class}")
    plt.show()

    return img, predicted_class


# Function called by the UI

def predict(image):
    model = load_model("../crop_health_monitoring_model.h5") 

    # path = "C:\\Users\\OMOLP094\\Desktop\\Research Projects\\Soham Jariwala - Crop Health Monitoring Project\\Crop Health Monitoring - Final Implementation\\test\\CornCommonRust1.JPG"

    path = image.name

    # Call the prediction function
    img, predicted_class = prediction(path, model, classes_dict_cleaned)

    return img, predicted_class


# User Interface

def main(): 
    io = gr.Interface(
        fn=predict,
        inputs=gr.File(label="Upload the image of the leaf of the plant or crop", file_types = ["image"]),
        outputs = [gr.Image(label = "Uploaded Image", width = 400, height = 400), 
                   gr.Textbox(label="Crop Health Overview")],
        allow_flagging="manual",
        flagging_options=["Save"],
        title="CNN-Powered Crop Health Monitoring System",
        description="Effortlessly monitor crop health with our CNN-powered system.",
        theme = gr.themes.Soft()
    )

    io.launch(share=True)

if __name__ == "__main__":
    main()



