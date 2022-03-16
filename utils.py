import numpy as np


# generate a guassian random variable with 1 standard deviation equal to 0.1x the input variable
def gaussian_rand(in_var):
    gaussian_random_offset = np.random.randn(1)[0]
    scaled_offset = gaussian_random_offset * 0.1 * in_var
    ret_val = in_var + scaled_offset
    if ret_val < in_var * 0.5:
        ret_val = in_var * 0.5
    elif ret_val > in_var * 1.5:
        ret_val = in_var * 1.5
    return ret_val
