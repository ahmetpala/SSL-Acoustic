import numpy as np

def db(data, eps=1e-10):
    """ Decibel (log) transform """
    return 10 * np.log10(data + eps)

def db_with_limits(data, limit_low=-75, limit_high=0):
    data = db(data)
    data[data>limit_high] = limit_high
    data[data<limit_low] = limit_low
    return data


def xr_db_with_limits(xarr, limit_low=-75, limit_high=0): # Function to be applied to xarray dataarray
    data = db(xarr)
    result = data.where((data<limit_high) | (data.isnull()), limit_high)
    result = result.where((result>limit_low) | (result.isnull()), limit_low)
    return result
