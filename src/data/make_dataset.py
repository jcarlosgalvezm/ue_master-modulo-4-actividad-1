# -*- coding: utf-8 -*-
import click
import logging
import torchvision
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[2]


def get_dataset(path):
    train = torchvision.datasets.MNIST(root=path, train=True, download=True,
        transform=None)
    test = torchvision.datasets.MNIST(root=path, train=False,
        download=True, transform=None)

    return train, test


@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    if not os.path.exists(output_filepath):
        click.ClickException('Path doesn\'t exists').show()
        return

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data...')
    get_dataset(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
