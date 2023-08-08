import os

from . import embedders
from .encoders import Encoder
from .decoders import Decoder
from .dsap import DSAP


def build_model(params):
    decoder_only = params["model"].get('decoder_only')
    if decoder_only:
        embedder = encoder = None
    else:
        embedder = embedders.RNNEmbed(params)
        encoder = Encoder(embedder, params)
    decoder = Decoder(params)
    model = DSAP(encoder, decoder, params)

    print("Model: ", model)
    if not params['no_gpu']:
        model.cuda()
    if params['load_best_model']:
        print("LOADING BEST MODEL")
        path = os.path.join(params['working_dir'], 'best_model')
        model.load(path)
    elif params['load_model']:
        print("LOADING MODEL FROM PATH: ", params['load_model'])
        model.load(params['load_model'])
    return model