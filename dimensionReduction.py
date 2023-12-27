from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from loguru import logger
import cv2


def calculate_mse_directly(original_pixels, reconstructed_pixels):
    """
    Calculate Mean Squared Error (MSE) directly between two sets of pixels.

    Parameters:
    - original_pixels (numpy.ndarray): Array containing the original pixel values.
    - reconstructed_pixels (numpy.ndarray): Array containing the reconstructed pixel values.

    Returns:
    - float: Calculated MSE between original and reconstructed pixels.
    """
    if original_pixels.shape != reconstructed_pixels.shape:
        raise ValueError("The images must have the same number of pixels.")

    # Calculate the number of pixels
    n_pixels = original_pixels.shape[0]

    # Direct calculation of MSE
    mse = np.sum((original_pixels - reconstructed_pixels) ** 2) / n_pixels

    return mse


def dimension_reduction(image_path):
    """
    Perform dimension reduction on a given image using Principal Component Analysis (PCA).

    Parameters:
    - image_path (str): Path to the input image file.

    Returns:
    - int: Minimum number of components required to achieve reconstruction error less than or equal to 3.0.
    """
    # Load the image
    image_bgr = cv2.imread(image_path)

    # Convert to grayscale
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Normalize the grayscale image
    normalized_image = gray_image / 255.0

    # Initialize the MSE threshold
    mse_threshold = 3.0

    # Initialize variables for finding the minimum number of components
    mse = float('inf')
    min_n = None
    reconstructed_image = None

    # Iterate from the minimum number of components to 1 to find the best number of components
    for n in range(min(gray_image.shape), 0, -1):
        # Perform PCA
        pca = PCA(n_components=n)
        pca.fit(normalized_image)  # Fit PCA on the image
        transformed_data = pca.transform(normalized_image)
        reconstructed_data = pca.inverse_transform(transformed_data)
        reconstructed_data_original_scale = reconstructed_data * 255.0
        # Calculate MSE
        mse = mean_squared_error(gray_image,
                                 reconstructed_data_original_scale)

        # If MSE is within the threshold, store the number of components and reconstructed image
        if mse <= mse_threshold:
            min_n = n
            reconstructed_image = reconstructed_data
            logger.info(f"MSE for n={n}: {mse}")
            continue  # Stop if we find a good reconstruction
        else:
            break

    # Plot the original, grayscale, and reconstructed images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original Image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Grayscale Image
    axes[1].imshow(gray_image, cmap='gray')
    axes[1].set_title('Gray Scale Image')
    axes[1].axis('off')

    # Reconstructed Image
    if reconstructed_image is not None:
        axes[2].imshow(reconstructed_image, cmap='gray')
        axes[2].set_title(f'Reconstructed Image (n={min_n})')
    else:
        axes[2].set_title("No reconstruction within MSE threshold")
    axes[2].axis('off')

    # Adjust layout to prevent overlapping titles
    plt.tight_layout(pad=2.0)

    # Show the plot
    plt.show()

    # Return the minimum number of components that met the MSE threshold
    return min_n
