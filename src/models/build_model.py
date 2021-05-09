import torch
from skorch import NeuralNetClassifier

DEFAULT_PARAMS = {
    'max_epochs': 25,
    'iterator_train__num_workers': 4,
    'iterator_valid__num_workers': 4,
    'lr': 0.001,
    'batch_size': 64,
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class NNetModule(torch.nn.Module):
    """ Convolutional Neuronal Network"""
    def __init__(self, network=None):
        super().__init__()
        self.network = network or torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128 * 7 * 7, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 10),
            torch.nn.LogSoftmax(dim=1)
        )

    def initialize(self):
        """ Initialize NN weights using xavier normal"""
        super().initialize()
        for nam, param in self.module_.named_parameters():
            if 'weight' in nam:
                torch.nn.init.xavier_normal_(param)
            else:
                param.data.fill_(0.01)

    def forward(self, X):
        return self.network(X)


def build_model(bucket, filename='model.pkl', **kwargs):
    if not kwargs:
        kwargs = DEFAULT_PARAMS

    model = NeuralNetClassifier(
        NNetModule,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.CrossEntropyLoss,
        warm_start=True,
        device=DEVICE,
        **kwargs
    )

    return model
