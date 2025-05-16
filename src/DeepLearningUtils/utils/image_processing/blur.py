
# https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python/17201686#17201686
# https://stackoverflow.com/questions/55643675/how-do-i-implement-gaussian-blurring-layer-in-keras
def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    h = h.astype(np.float32)
    return h
