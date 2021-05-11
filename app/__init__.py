import pickle
import logging

from ibm_cloud_sdk_core import ApiException
from ibmcloudant.cloudant_v1 import Document

from flask import Flask, current_app
from flask_restx import Api
from sklearn.pipeline import make_pipeline
from src.vendor.IBM import cloudant, cos
from src.models.build_model import build_model, DEFAULT_PARAMS
from src.features.build_features import Scaler, Unsqueeze
from src.data.make_dataset import get_dataset


api = Api(
    title='MNIST prediction API',
    description='Master Online IA y Data Science 2020-2021 '
    '- Módulo 4 Ciclo de Vida de los Modelos - Actividad evaluativa - '
    'Juan Carlos Gálvez Martínez'
    )


def setup_model():
    cloudant_client = cloudant.get_client('MODEL_CATALOG')
    cos_client = cos.get_client()
    bucket = 'object-storage-mnist-model'

    try:
        cos_client.create_bucket(Bucket=bucket)
    except cos_client.exceptions.BucketAlreadyExists:
        pass

    cloudant.create_db(cloudant_client, 'models')
    cloudant.create_db(cloudant_client, 'predictions')

    model = None

    try:
        result = cloudant_client.get_document(db='models',
            doc_id='mnist').get_result()
        rev = result['_rev'].split('-', 1)[0]
        fname = f'model_v{rev}.pkl'
    except ApiException:
        # Doesn't exists documents, we need to create the first one
        doc = Document(id='mnist', **DEFAULT_PARAMS)
        result = cloudant_client.post_document(db='models', document=doc)
        rev = result['rev'].split('-', 1)[0]
        fname = f'model_v{rev}.pkl'
        try:
            # Delete legacy cos object to mark it as it needs to be trained
            cos_client.delete_object(Bucket=bucket, Key=fname)
        except cos_client.exceptions.NoSuchKey:
            pass

    try:
        model = cos_client.get_object(Bucket=bucket, Key=fname)
        current_app.logger.info(f'Latest model v{rev} loaded...')
    except cos_client.exceptions.NoSuchKey:
        project_dir = current_app.config['PROJECT_DIR']
        train_ds, test_ds = get_dataset(f'{project_dir}/data')
        model = build_model(bucket=bucket)
        model.initialize()
        pipeline = make_pipeline(Scaler(), Unsqueeze(), model)
        pipeline.fit(train_ds.data[:].float(), train_ds.targets[:])
        cos_client.put_object(Bucket=bucket, Key=fname,
            Body=pickle.dumps(model))
        current_app.logger.info(f'Latest model v{rev} created and loaded...')


def create_app():
    app = Flask(__name__)

    if app.config['ENV'] == 'development':
        app.config.from_pyfile('config.py')
    else:
        app.config.from_pyfile('config.prod.py')

    @app.before_first_request
    def warmup():
        with app.app_context():
            setup_model()

    from app.api_factory import make_namespaces

    api.init_app(app)
    for ns in make_namespaces(api):
        api.add_namespace(ns)

    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

    return app
