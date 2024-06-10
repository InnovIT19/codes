import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
import matplotlib.pyplot as plt

# RGB Color Space Segmentation
def rgb_skin_segmentation(image):
    rule1 = (image[:, :, 0] > 95) & (image[:, :, 1] > 40) & (image[:, :, 2] > 20) & \
            ((np.max(image, axis=2) - np.min(image, axis=2)) > 15) & \
            (np.abs(image[:, :, 0] - image[:, :, 1]) > 15) & (image[:, :, 0] > image[:, :, 1]) & \
            (image[:, :, 0] > image[:, :, 2])

    rule2 = (image[:, :, 0] > 220) & (image[:, :, 1] > 210) & (image[:, :, 2] > 170) & \
            (np.abs(image[:, :, 0] - image[:, :, 1]) <= 15) & (image[:, :, 2] < image[:, :, 0]) & \
            (image[:, :, 2] < image[:, :, 1])

    mask = rule1 | rule2
    return mask

# YCbCr Color Space Segmentation
def ycbcr_skin_segmentation(image):
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = cv2.split(ycbcr_image)
    
    mask = (Cr <= 1.5862 * Cb + 20) & (Cr >= 0.3448 * Cb + 76.2069) & \
           (Cr >= -4.5652 * Cb + 234.5652) & (Cr <= -1.15 * Cb + 301.75) & \
           (Cr <= -2.2857 * Cb + 432.85)
    
    return mask

# HSV Color Space Segmentation
def hsv_skin_segmentation(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv_image)
    
    mask = (H < 50) | (H > 150)
    return mask

# Combined Segmentation
def combined_skin_segmentation(image):
    rgb_mask = rgb_skin_segmentation(image)
    ycbcr_mask = ycbcr_skin_segmentation(image)
    hsv_mask = hsv_skin_segmentation(image)
    
    combined_mask = rgb_mask & ycbcr_mask & hsv_mask
    return combined_mask

# Classify Skin Tone
def classify_skin_tone(image, mask):
    skin_pixels = image[mask]
    kmeans = KMeans(n_clusters=5, random_state=0).fit(skin_pixels)
    skin_tone = kmeans.cluster_centers_.astype(int)
    return skin_tone

# Extract skin pixels and calculate mean RGB value
def extract_skin_pixels(image):
    mask = combined_skin_segmentation(image)
    skin_pixels = image[mask]
    return skin_pixels

# Determine closest skin tone
def determine_closest_skin_tone(mean_skin_rgb, centroids):
    distances = distance.cdist([mean_skin_rgb], centroids, 'euclidean')
    closest_index = np.argmin(distances)
    return closest_index

# Centroids
centroids = np.array([[165, 109, 79], [252, 234, 203], [205, 149, 117], [122, 68, 43], [242, 188, 153]])
skin_tones = ['Medium Brown', 'Fair', 'Light Brown', 'Dark Brown', 'Light to Medium Brown']

# Identify the skin tone from the image path
def identify_skin_tone(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = combined_skin_segmentation(image)
        skin_tone = classify_skin_tone(image, mask)
        if skin_tone is not None:
            print("Skin Tone RGB Values:", skin_tone)
            
            # Calculate mean RGB value of the skin pixels
            skin_pixels = extract_skin_pixels(image)
            mean_skin_rgb = np.mean(skin_pixels, axis=0)
            
            # Determine closest skin tone
            closest_index = determine_closest_skin_tone(mean_skin_rgb, centroids)
            predicted_skin_tone = skin_tones[closest_index]
            
            print(f'The skin tone of the image is: {predicted_skin_tone}')
            
            # Display the segmented skin
            skin_segmented = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
            plt.figure(figsize=(10, 10))
            plt.imshow(skin_segmented)
            plt.title("Skin Segmentation")
            plt.axis('off')
            plt.show()
        else:
            print("No skin tone detected. Try another image.")
    else:
        print(f"Error: Unable to read image at {image_path}")

# Sample
image_path = input("Enter the path of the image: ").strip()
# Raw string to handle backslashes in the file path
image_path = r'{}'.format(image_path)
identify_skin_tone(image_path)
