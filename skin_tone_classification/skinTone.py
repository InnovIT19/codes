from tensorflow.keras.models import load_model

# Load the model
model = load_model("skin_tone_classification_model_v2.h5")

# Function to preprocess a single image and make a prediction
def classify_skin_tone(image_path, model, img_size):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_normalized = img_resized / 255.0
    img_reshaped = np.reshape(img_normalized, (1, img_size, img_size, 3))
    prediction = model.predict(img_reshaped)
    predicted_class = np.argmax(prediction)
    return categories[predicted_class]

# Example usage
image_path = "path_to_new_image"
predicted_tone = classify_skin_tone(image_path, model, img_size)
print(f"The predicted skin tone is: {predicted_tone}")
