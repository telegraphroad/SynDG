import torch
import torch.nn as nn

def tukey_biweight_estimator(tensor, initial_guess=None, c=1.345, max_iter=100, tol=1e-6):
    """
    Compute the Huber M-estimator for a 1D tensor.

    Parameters:
    - tensor: 1D tensor, the input data
    - initial_guess: float, initial guess for the estimator
    - c: float, tuning constant (default is 1.345)
    - max_iter: int, maximum number of iterations (default is 100)
    - tol: float, convergence tolerance (default is 1e-6)

    Returns:
    - mu: float, the Huber M-estimator of the data
    """
    if initial_guess is None:
        mu = tensor.median()
    else:
        mu = initial_guess

    for _ in range(max_iter):
        diffs = tensor - mu
        weights = torch.where(torch.abs(diffs) <= c, torch.ones_like(tensor), c / torch.abs(diffs))
        mu_next = torch.sum(weights * tensor) / torch.sum(weights)

        if torch.abs(mu - mu_next) < tol:
            break

        mu = mu_next

    return mu


def geometric_median_of_means_pyt(samples, num_buckets, max_iter=100, eps=1e-5):
    """
    Compute the geometric median of means by placing `samples`
    in num_buckets using Weiszfeld's algorithm
    """
    if len(samples.shape) == 1:
        samples = samples.reshape(-1, 1)
    bucketed_means = torch.stack(
        [torch.mean(val, dim=0) for val in torch.split(samples, num_buckets)]
    )

    if bucketed_means.shape[0] == 1:
        return bucketed_means.squeeze()  # when sample size is 1, the only sample is the median
    print('pyt median')

    # This reduces the chance that the initial estimate is close to any
    # one of the data points
    gmom_est = torch.mean(bucketed_means, dim=0)

    for i in range(max_iter):
        weights = 1 / torch.norm(bucketed_means - gmom_est, dim=1, p=2)[:, None]
        old_gmom_est = gmom_est
        gmom_est = (bucketed_means * weights).sum(dim=0) / weights.sum()
        if (
            torch.norm(gmom_est - old_gmom_est, p=2) / torch.norm(old_gmom_est, p=2)
            < eps
        ):
            break

    return gmom_est