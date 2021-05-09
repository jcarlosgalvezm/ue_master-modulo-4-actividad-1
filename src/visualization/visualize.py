import numpy as np
import matplotlib.pyplot as plt


def plot_img_sample_grid(data, size: int, labels: np.ndarray):
    nrows = ncols = size
    nsamples = nrows * ncols
    for i in range(nsamples):
        plt.subplot(nrows, ncols, (i + 1))
        plt.imshow(data[i], cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.title(labels[i])
    plt.show()
