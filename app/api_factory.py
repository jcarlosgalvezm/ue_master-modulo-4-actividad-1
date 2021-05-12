import io
import pickle

from flask import current_app
from flask_restx import Namespace, Resource
from PIL import Image
from sklearn.pipeline import make_pipeline
from src.vendor.IBM import cloudant, cos
from src.data.make_dataset import get_dataset
from src.features.build_features import Preprocess, Scaler, Unsqueeze
from werkzeug.datastructures import FileStorage


#cat = api.model('Cat', {
#    'id': fields.String(required=True, description='The cat identifier'),
#    'name': fields.String(required=True, description='The cat name'),
#})

#CATS = [
#    {'id': 'felix', 'name': 'Felix'},
#]


def make_namespaces(api):

    cloudant_client = cloudant.get_client('MODEL_CATALOG')
    cos_client = cos.get_client()
    bucket = 'object-storage-mnist-model'

    result = cloudant_client.get_document(db='models',
        doc_id='mnist', revs=True).get_result()

    for num, rev in enumerate(result['_revisions']['ids'][::-1], start=1):
        try:
            fname = f'model_v{num}.pkl'
            ns = Namespace(f'v{num}', description=f'MNIST_v{num}')
            upload_parser = ns.parser()
            upload_parser.add_argument('image', location='files',
                type=FileStorage, required=True)
            cos_client.get_object(Bucket=bucket, Key=fname)

            @ns.route('/info')
            class Info(Resource):

                FNAME = fname
                BUCKET = bucket
                REVISION = f'{num}-{rev}'

                def get(self):
                    '''Model info'''
                    project_dir = current_app.config['PROJECT_DIR']
                    train_ds, test_ds = get_dataset(f'{project_dir}/data')
                    X_test, y_test= test_ds.data[:].float(), test_ds.targets[:]

                    model_buf = io.BytesIO()
                    cos_client.download_fileobj(Bucket=self.BUCKET,
                        Key=self.FNAME, Fileobj=model_buf)
                    model_buf.seek(0)
                    model = pickle.load(model_buf)

                    pipeline = make_pipeline(
                        Scaler(),
                        Unsqueeze(),
                        model
                        )
                    score = pipeline.score(X_test, y_test)
                    return {
                        'definition': cloudant_client.get_document(
                                db='models',
                                doc_id='mnist',
                                rev=self.REVISION).get_result(),
                        'score': score
                        }

            @ns.route('/predict')
            @ns.expect(upload_parser)
            class Predict(Resource):

                FNAME = fname
                BUCKET = bucket

                def post(self):
                    '''Predict number'''
                    args = upload_parser.parse_args()

                    model_buf = io.BytesIO()
                    cos_client.download_fileobj(Bucket=self.BUCKET,
                        Key=self.FNAME, Fileobj=model_buf)
                    model_buf.seek(0)
                    model = pickle.load(model_buf)

                    img_buf = io.BytesIO()
                    args['image'].save(img_buf)
                    uploaded_img = Image.open(img_buf)

                    pipeline = make_pipeline(
                        Preprocess(),
                        Scaler(),
                        Unsqueeze(),
                        model
                        )

                    X = [uploaded_img]
                    y_hat = pipeline.predict(X)

                    return {
                        'result': int(y_hat[0])
                    }

            yield ns

        except cos_client.exceptions.NoSuchKey:
            pass
