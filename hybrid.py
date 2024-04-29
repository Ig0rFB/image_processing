import numpy as np
import cv2

image1 = cv2.imread('imgs/helen.png')
image2 = cv2.imread('imgs/john.png')

image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


def _cross_correlation_2d_channel(channel, kernel):
    # Get dimensions and calculate padding
    img_h, img_w = channel.shape
    k_h, k_w = kernel.shape

    pad_h = k_h // 2
    pad_w = k_w // 2

    padded_img = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(channel)
    

    for i in range(img_h):
        for j in range(img_w):
            output[i, j] = np.sum(padded_img[i:i+k_h, j:j+k_w] * kernel)
    
    return output

def cross_correlation_2d(img, kernel):
    if len(img.shape) == 3:  
        r = _cross_correlation_2d_channel(img[:, :, 0], kernel)
        g = _cross_correlation_2d_channel(img[:, :, 1], kernel)
        b = _cross_correlation_2d_channel(img[:, :, 2], kernel)
        return np.stack([r, g, b], axis=-1)
    else:  
        return _cross_correlation_2d_channel(img, kernel)

def convolve_2d(img, kernel):

    flipped_kernel = np.flip(kernel)
    return cross_correlation_2d(img, flipped_kernel)

def gaussian_blur_kernel_2d(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma**2)) * 
                     np.exp(- ((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2 * sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def low_pass(img, size, sigma):
    kernel = gaussian_blur_kernel_2d(size, sigma)
    return convolve_2d(img, kernel)

def high_pass(img, size, sigma):
    return img - low_pass(img, size, sigma)

def create_hybrid_image(image1, image2, size, sigma):
    low_pass_img1 = low_pass(image1, size, sigma)
    high_pass_img2 = high_pass(image2, size, sigma)
    
    # Combine the two images
    hybrid_image = low_pass_img1 + high_pass_img2
    hybrid_image = np.clip(hybrid_image, 0, 255).astype(np.uint8)    
    return hybrid_image


hybrid = create_hybrid_image(image1, image2, size=15, sigma=7)

cv2.imwrite('hybrid_image.jpg', cv2.cvtColor(hybrid, cv2.COLOR_RGB2BGR))
