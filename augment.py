import cv2
import os
import pandas as pd
import numpy as np
import random

# --- Augmentation Functions ---
n = 250


def add_noise(image):
    """Adds random Gaussian noise to an image."""
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row, col = image.shape
    mean = 0
    var = 10
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def change_brightness(image, value=30):
    """Changes the brightness of an image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def flip_image(image):
    """Flips an image horizontally."""
    return cv2.flip(image, 1)

def rotate_image(image, angle):
    """Rotates an image by a given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def crop_image(image, scale=0.8):
    """Crops an image to a certain scale and resizes it back."""
    (h, w) = image.shape[:2]
    ch, cw = int(h * scale), int(w * scale)
    y = random.randint(0, h - ch)
    x = random.randint(0, w - cw)
    cropped = image[y:y+ch, x:x+cw]
    return cv2.resize(cropped, (w, h))

def stretch_image(image, sx, sy):
    """Stretches an image with given scaling factors."""
    (h, w) = image.shape[:2]
    return cv2.resize(image, (int(w*sx), int(h*sy)))

def zoom_image(image, scale=1.2):
    """Zooms into the center of an image."""
    (h, w) = image.shape[:2]
    # Crop center
    ch, cw = int(h / scale), int(w / scale)
    y = (h - ch) // 2
    x = (w - cw) // 2
    zoomed = image[y:y+ch, x:x+cw]
    return cv2.resize(zoomed, (w, h))

def change_rgb_channels(image):
    """Randomly shuffles the RGB channels of an image."""
    channels = list(cv2.split(image))
    random.shuffle(channels)
    return cv2.merge(channels)

def change_contrast(image, alpha=1.5):
    """Changes the contrast of an image."""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def sharpen_image(image):
    """Sharpens an image using a kernel."""
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

def blur_image(image, ksize=(5,5)):
    """Blurs an image using a Gaussian filter."""
    return cv2.GaussianBlur(image, ksize, 0)

def random_erasing(image, p=0.5, sl=0.02, sh=0.4, r1=0.3):
    """Performs random erasing."""
    if random.uniform(0, 1) > p:
        return image

    img = image.copy()
    (h, w) = img.shape[:2]
    area = h * w

    for _ in range(100):
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1/r1)

        H = int(round(np.sqrt(target_area * aspect_ratio)))
        W = int(round(np.sqrt(target_area / aspect_ratio)))

        if W < w and H < h:
            x1 = random.randint(0, h - H)
            y1 = random.randint(0, w - W)
            img[x1:x1+H, y1:y1+W] = np.random.uniform(0, 255)
            return img
    return image


# --- Augmentation Process ---

dataset_path = "C:\python_test\ML_project\data"
MIN_IMAGES_PER_LABEL = n 

print("Starting data augmentation...")

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if not os.path.isdir(label_path):
        continue

    images = [f for f in os.listdir(label_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(images)

    if num_images < MIN_IMAGES_PER_LABEL:
        num_to_generate = MIN_IMAGES_PER_LABEL - num_images
        print(f"Label {label}: Found {num_images} images. Generating {num_to_generate} more.")
        
        image_files_to_augment_from = list(images)

        if not image_files_to_augment_from:
            print(f"Warning: Label {label} has no images to augment from. Skipping.")
            continue

        for i in range(num_to_generate):
            random_image_name = random.choice(image_files_to_augment_from)
            random_image_path = os.path.join(label_path, random_image_name)
            img = cv2.imread(random_image_path)
            
            if img is None:
                print(f"Warning: Could not read image {random_image_path}. Skipping.")
                continue

            # List of available augmentations
            augmentations = [
                'noise', 'brightness', 'flip', 'rotate', 'crop', 'stretch', 
                'zoom', 'contrast', 'sharpen', 'blur', 'random_erasing'
            ]
            augmentation = random.choice(augmentations)

            augmented_img = None
            if augmentation == 'noise':
                # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # noisy_gray = add_noise(gray_img)
                # augmented_img = cv2.cvtColor(noisy_gray, cv2.COLOR_GRAY2BGR)
                pass
            elif augmentation == 'brightness':
                augmented_img = change_brightness(img, random.randint(-40, 40))
            elif augmentation == 'flip':
                augmented_img = flip_image(img)
            # elif augmentation == 'rotate':
            #     augmented_img = rotate_image(img, random.randint(-25, 25))
            elif augmentation == 'crop':
                augmented_img = crop_image(img, scale=random.uniform(0.7, 0.95))
            elif augmentation == 'stretch':
                augmented_img = stretch_image(img, sx=random.uniform(0.8, 1.2), sy=random.uniform(0.8, 1.2))
            elif augmentation == 'zoom':
                augmented_img = zoom_image(img, scale=random.uniform(1.1, 1.5))
            # elif augmentation == 'rgb_channels':
            #     augmented_img = change_rgb_channels(img)
            elif augmentation == 'contrast':
                augmented_img = change_contrast(img, alpha=random.uniform(1.1, 1.5))
            elif augmentation == 'sharpen':
                augmented_img = sharpen_image(img)
            elif augmentation == 'blur':
                augmented_img = blur_image(img, ksize=(random.choice([3,5,7]), random.choice([3,5,7])))
            elif augmentation == 'random_erasing':
                # augmented_img = random_erasing(img)
                pass
            else:
                pass


            if augmented_img is not None:
                # The stretched image might have a different size, so we resize it back
                if augmented_img.shape[:2] != img.shape[:2]:
                     augmented_img = cv2.resize(augmented_img, (img.shape[1], img.shape[0]))

                new_filename = f"aug_{i}_{random_image_name}"
                new_image_path = os.path.join(label_path, new_filename)
                cv2.imwrite(new_image_path, augmented_img)

print("Data augmentation finished.")
print("-" * 20)
print("Starting CSV creation process...")


# Define the path to the main dataset directory
# This is already defined above, but we'll keep the script structure
output_path = "c:\\python_test\\ML\\templates\\dataset.csv"

data = []
labels = []

# Iterate over each sub-directory (each representing a label)
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if not os.path.isdir(label_path):
        continue

    # Iterate over each image in the sub-directory
    for filename in os.listdir(label_path):
        if filename.endswith(".jpg"):
            # Read the image
            img_path = os.path.join(label_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Warning: Could not read image {img_path} during CSV creation. Skipping.")
                continue

            # Resize the image to a fixed size (e.g., 50x50)
            img_resized = cv2.resize(img, (50, 50))
            
            # Flatten the image into a 1D array
            img_flattened = img_resized.flatten()
            
            # Append the flattened image data and the label
            data.append(img_flattened)
            labels.append(label)

# Create a DataFrame
df = pd.DataFrame(data)
df['label'] = labels

# Save the DataFrame to a CSV file
df.to_csv(output_path, index=False)

print(f"Dataset created and saved to {output_path}")
print(f"Total images processed: {len(df)}")
if __name__ == "__main_":
    print(df)
print(df)

