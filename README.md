MNIST_ue
==============================

UE AI Master activity :: Author - Juan Carlos Gálvez Martínez


CNN web service based on python Flask

------------
# Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── app                <- REST api tha serves model actions
    │
    ├── data               <- Data source files.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── build_model.py
    │   │
    │   ├── vendor         <- Vendor drivers
    │   │   │
    │   │   └── IBM
    │   │       ├── cloudant.py
    │   │       └── cos.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

# Instructions

## Set enviromnet variables

You must create an .env file on the project root directory. You can use the .env.template file for this purpose, setting your IBM Cloud credential values.

## Deployment

### Using Flask:

For development purpose you can launch the webapp typing:

```{bash}
$ flask run
```
The webapp will published at *127.0.0.1:5000*


### Using Gunicorn:

For production purpose you can launch the webapp typing:

```{bash}
$ gunicorn --bind 0.0.0.0:8899 --timeout 12000 wsgi:app
```

The webapp will published at *0.0.0.0:8899*

### Using Docker:

Build the docker image typing:

```{bash}
$ docker build -t mnist_app .
```
*(don’t forget the dot)*

Run the docker image:
```{bash}
docker run -it -d -p 8899:8899 mnist_app
```

The webapp will published at *0.0.0.0:8899*
