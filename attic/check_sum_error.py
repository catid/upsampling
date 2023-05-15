import random
import math

def compute_mean_std(num_pixels, use_compensated_summation=False):
    # Generate random pixel values
    pixel_values = [random.random() for i in range(num_pixels)]
    
    # Compute the sum of the pixel values
    sum_pixels = 0
    error_term = 0
    for pixel_value in pixel_values:
        if use_compensated_summation:
            y = pixel_value - error_term
            t = sum_pixels + y
            error_term = (t - sum_pixels) - y
            sum_pixels = t
        else:
            sum_pixels += pixel_value

    # Compute the mean of the pixel values
    mean = sum_pixels / num_pixels
    
    # Compute the sum of the squared pixel values
    sum_squared_pixels = 0
    error_term = 0
    for pixel_value in pixel_values:
        if use_compensated_summation:
            y = pixel_value**2 - error_term
            t = sum_squared_pixels + y
            error_term = (t - sum_squared_pixels) - y
            sum_squared_pixels = t
        else:
            sum_squared_pixels += pixel_value**2
    
    # Compute the standard deviation of the pixel values
    std = math.sqrt((sum_squared_pixels / num_pixels) - (mean**2))

    return mean, std

# Compute the mean and standard deviation of 1 million pixel values using naïve summation
mean_naive, std_naive = compute_mean_std(1000000)

# Compute the mean and standard deviation of 1 million pixel values using compensated summation
mean_compensated, std_compensated = compute_mean_std(1000000, use_compensated_summation=True)

# Print the results
print(f"Naïve summation: Mean = {mean_naive}, Std = {std_naive}")
print(f"Compensated summation: Mean = {mean_compensated}, Std = {std_compensated}")
