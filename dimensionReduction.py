from skimage import io, color
from skimage.transform import resize
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function for dimension reduction and plotting
def dimension_reduction(image_path):
    # Load the image
    image = io.imread(image_path)
    # Convert to grayscale
    gray_image = color.rgb2gray(image)
    # Normalize the grayscale image
    normalized_image = gray_image / 255.0
    # Flatten the image to 1D array for PCA
    flatten_image = normalized_image.flatten()
    
    # Initialize PCA with minimum possible components
    pca = PCA(n_components=min(normalized_image.shape))
    # Fit and transform the image data
    transformed_data = pca.fit_transform(normalized_image)
    
    # Find the minimum number of components with reconstruction error less or equal to 3.0
    min_error = float('inf')
    n_components = 0
    reconstruction = None
    
    for i in range(1, transformed_data.shape[1] + 1):
        # Apply inverse transform to reconstruct the image
        pca.n_components = i
        print(i)
        reconstructed = pca.inverse_transform(transformed_data)
        # Calculate MSE
        mse = np.mean((flatten_image - reconstructed.flatten()) ** 2)
        # Check if the reconstruction error is less than or equal to 3.0
        if mse <= 3.0 and mse < min_error:
            min_error = mse
            n_components = i
            reconstruction = reconstructed
            break
    
    # Reshape the reconstructed image back to its original shape
    reconstructed_image = reconstruction.reshape(normalized_image.shape)
    
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
    axes[2].imshow(reconstructed_image, cmap='gray')
    axes[2].set_title(f'Reconstruction (n={n_components})')
    axes[2].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Return the number of components used for reconstruction
    return n_components

