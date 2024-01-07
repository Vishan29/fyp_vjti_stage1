import cv2
import numpy as np

def spatial_hierarchical_pooling_image(image_path, pool_size=(1, 1)):
    # Read the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get the dimensions of the image
    height, width = image.shape

    # Check if the image dimensions are divisible by the pool size
    assert height % pool_size[0] == 0, "Invalid pool size for height"
    assert width % pool_size[1] == 0, "Invalid pool size for width"

    # Initialize the pooled image
    pooled_image = np.zeros((height // pool_size[0], width // pool_size[1]))

    # Iterate through the image with the specified pool size
    for i in range(0, height, pool_size[0]):
        for j in range(0, width, pool_size[1]):
            # Take the maximum value within the pool size
            pooled_image[i // pool_size[0], j // pool_size[1]] = np.max(image[i:i+pool_size[0], j:j+pool_size[1]])

    return pooled_image

# Example usage:
# Specify the path to your image
image_path = 'UCMerced_LandUse/Images/agricultural/agricultural00.tif'

# Apply spatial hierarchical pooling with a pool size of (2, 2)
pooled_image = spatial_hierarchical_pooling_image(image_path, pool_size=(1, 1))

# Display the original and pooled images
cv2.imshow('Original Image', cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
cv2.imshow('Pooled Image', pooled_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
