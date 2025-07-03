from nphm.models.setup_training import set_up_networks
from nphm.models.neural3dmm import nn3dmm
from nphm.nphmoctree import NPHMOctree

def setup_nphm_grid(cfg, load_dict, encoding, anchors, aabb_min, aabb_max, scale=4.0, resolution=128, rank="cuda"):
    args = {"model_type": "nphm"}
    id_decoder, ex_decoder = set_up_networks(cfg,
        anchors=anchors,
        rank=rank,
        load_dict=load_dict,
        **args
    )
    nphm = nn3dmm(
        id_model=id_decoder,
        ex_model=ex_decoder,
        expr_direction='backward'
    ).to(rank)

    nphm_grid = NPHMOctree(
        decoder=nphm,
        encoding=encoding,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        scale=scale,
        resolution=resolution,
        device=rank
    )
    return nphm_grid
