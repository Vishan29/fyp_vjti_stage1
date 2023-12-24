import os
import cv2
import numpy as np

def extract_patches(image, patch_size):
    height, width = image.shape[:2]
    patches = []
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches

def process_images(input_folder, output_folder, patch_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    categories = os.listdir(input_folder)
    
    for category in categories:
        category_path = os.path.join(input_folder, category)
        output_category_path = os.path.join(output_folder, category)
        
        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        image_files = [f for f in os.listdir(category_path) if f.endswith('.tif')]

        for image_file in image_files:
            image_path = os.path.join(category_path, image_file)
            output_image_path = os.path.join(output_category_path, os.path.splitext(image_file)[0])

            # Create output directory for the image
            if not os.path.exists(output_image_path):
                os.makedirs(output_image_path)

            # Read the image
            image = cv2.imread(image_path)

            # Extract patches
            patches = extract_patches(image, patch_size)

            # Save patches
            for idx, patch in enumerate(patches):
                patch_filename = f"patch{idx+1}.png"
                patch_output_path = os.path.join(output_image_path, patch_filename)
                cv2.imwrite(patch_output_path, patch)

if __name__ == "__main__":
    input_folder = "UCMerced_LandUse/Images"
    output_folder = "patched_images"
    patch_size = 64  # Adjust the patch size as needed

    process_images(input_folder, output_folder, patch_size)
