import os
from typing import OrderedDict
from data.data_loader import CreateDataLoader
from models.models import create_model
from util import visualizer
from util.visualizer import Visualizer
import util.util as util
from flask import Flask
from options.test_options import TestOptions
import torch

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


def get_model_options():
    model_options = TestOptions().parse(save=False)
    model_options.nThreads = 1
    model_options.batchSize = 1
    model_options.serial_batches = True
    model_options.no_flip = True
    model_options.verbose = os.environ["VERBOSE"]
    return model_options


pix2pixModel = None

def load_model():
    model_options = get_model_options()
    # data_loader = CreateDataLoader(model_options)
    # dataset = data_loader.load_data()
    visualizer = Visualizer(model_options)
    global pix2pixModel
    if(pix2pixModel != None):
        return pix2pixModel, model_options
    # create a model with test options
    _model = create_model(opt=model_options)
    if(model_options.verbose):
        print(_model)
    # set pix2pixModel
    pix2pixModel = _model
    return pix2pixModel, model_options, visualizer



def generate_inference(dataset):
    _model, opt, visualizer = load_model()
    for i, data in enumerate(dataset):
        generated = _model.inference(
            data['label'], data['inst'], data['image'])
        visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                               ('synthesized_image', util.tensor2im(generated.data[0]))])
        img_path = data['path']
        print('Process image...... %s' % img_path)
        visualizer.save_raw_images(visuals, img_path, opt.results_dir)


def init():
    load_model()
