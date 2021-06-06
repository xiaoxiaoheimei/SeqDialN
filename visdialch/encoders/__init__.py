from visdialch.encoders.lf import LateFusionEncoder
from visdialch.encoders.bf import BlockFusionEncoder
from visdialch.encoders.dense import DenseEncoder


def Encoder(model_config, *args):
    name_enc_map = {"lf": LateFusionEncoder, "bf":BlockFusionEncoder, "dense":DenseEncoder}
    return name_enc_map[model_config["encoder"]](model_config, *args)
