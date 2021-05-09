import numpy as np
import torch
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin


class Preprocess(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        def transform_image(im):
            desired_size = (28, 28)
            im = im.convert('L')
            old_size = im.size
            ratio = float(desired_size[0]) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            height = width = (desired_size[0] - new_size[0]) // 2

            im = im.resize(new_size)

            new_im = Image.new("L", desired_size, color='white')
            new_im.paste(im, (height, width))
            return new_im

        imgarr = np.array([np.array(transform_image(im)) for im in X])

        return torch.from_numpy(imgarr).float()


class Scaler(TransformerMixin, BaseEstimator):
    """Scale dataset in range [-0.5, 0.5]"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X / X.max()) - 0.5

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class Unsqueeze(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.unsqueeze(1)
