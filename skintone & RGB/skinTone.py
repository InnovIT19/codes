import numpy as np
import pyodbc  # Assuming you are using pyodbc for SQL Server
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans

# Define the categories and image size
categories = ["dark", "light", "mid-dark", "mid-light"]
img_size = 64  # This should match the size used during training

# Load the trained model
model = load_model("skin_tone_classification_model_v2.h5")


def fetch_image_from_db(image_id):
    # Example function to fetch image blob from SQL Server
    conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                          'SERVER=DESKTOP-16A7BAU;'
                          'DATABASE=fashion;'
                          'UID=cube_sl;'
                          'PWD=123')
    cursor = conn.cursor()
    cursor.execute(f"SELECT Photo FROM Photos WHERE id = {image_id}")
    row = cursor.fetchone()
    image_data = row[0] if row else None
    conn.close()
    return image_data


def classify_skin_tone_from_db(image_id, model, img_size):
    image_data = fetch_image_from_db(image_id)

    if image_data is None:
        raise ValueError(f"Image with ID {image_id} not found in the database.")

    # Convert image data from blob to np.ndarray
    img_array = np.asarray(bytearray(image_data), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Resize and normalize image
    img_resized = cv2.resize(img, (img_size, img_size))
    img_normalized = img_resized / 255.0
    img_reshaped = np.reshape(img_normalized, (1, img_size, img_size, 3))

    # Perform prediction
    prediction = model.predict(img_reshaped)
    predicted_class = np.argmax(prediction)

    return categories[predicted_class], img


def classify_skin_tone(image, mask):
    skin_pixels = image[mask]
    kmeans = KMeans(n_clusters=5, random_state=0).fit(skin_pixels)
    skin_tone = kmeans.cluster_centers_.astype(int)
    return skin_tone


def create_skin_mask(image):
    # Convert to HSV color space for better skin color detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a binary mask where skin color range is detected
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Refine the mask using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

    return skin_mask > 0  # Convert to boolean mask


image_id = 2  # Replace with the ID of the image you want to classify
predicted_tone, img = classify_skin_tone_from_db(image_id, model, img_size)

# Create skin mask and classify skin tones
mask = create_skin_mask(img)
skin_tone_rgb_values = classify_skin_tone(img, mask)

print("Skin Tone RGB values:")
for idx, rgb in enumerate(skin_tone_rgb_values):
    print(f"Tone {idx + 1}: R={rgb[2]}, G={rgb[1]}, B={rgb[0]}")

print(f"The predicted skin tone is: {predicted_tone}")

# Display the image with mask
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title(f"Predicted Skin Tone: {predicted_tone}")
plt.axis('off')
plt.show()
