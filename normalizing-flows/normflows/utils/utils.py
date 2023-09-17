import torch

def tukey_biweight_estimator(tensor, max_iter=100, tol=1e-6):
    """
    This function calculates the Tukey's Biweight (or bisquare) estimator for a given tensor.
    Tukey's Biweight estimator is a robust statistic providing a resistant measure of central tendency.
    It's less sensitive to outliers compared to the mean or median.
    
    For more information, see:
    - Wikipedia: https://en.wikipedia.org/wiki/Bisquare_estimation
    
    Parameters:
    - tensor: The input tensor.
    - max_iter: The maximum number of iterations for the weight adjustment process (default: 100).
    - tol: A small threshold for the convergence of the estimator (default: 1e-6).
    
    Returns:
    - median: The Tukey's Biweight estimator of the input tensor.
    """
    
    # Initialize the median as the actual median of the tensor
    median = tensor.median()
    
    # Iterate to adjust the weights and update the median until the change in the median from one iteration to the next is smaller than a small tolerance value
    for _ in range(max_iter):
        # Make a copy of the current median
        prev_median = median.clone()
        
        # Compute the residuals as the difference between the tensor and the median
        diff = tensor - median
        
        # Compute the Median Absolute Deviation (MAD) of the residuals. MAD is a robust measure of scale (similar to standard deviation for normal distributions).
        mad = torch.median(torch.abs(diff))
        
        # Compute the standardized residuals (u-values) by dividing them by 6 times the MAD. This is in line with the usual practice for Tukey's Biweight estimator, where a value of 6 for the tuning constant provides approximately 95% efficiency for normally distributed data. 
        u = diff / (6.0 * mad)
        
        # Identify the residuals that are within the 6 MAD range
        mask = (torch.abs(u) <= 1.0)
        
        # Compute the bisquare weights for the residuals within the 6 MAD range
        weights = (1 - u[mask].square()) ** 2
        
        # Normalize the weights so they add up to 1
        weights /= weights.sum()
        
        # Compute the weighted average of the tensor values within the 6 MAD range
        median = (weights * tensor[mask]).sum()
        
        # If the change in the median is smaller than the tolerance, stop iterating
        if torch.abs(prev_median - median) < tol:
            break
    
    # Return the Tukey's Biweight estimator
    return median
