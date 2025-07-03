import numpy as np
import torch

# NNs
from nphm.models.canonical_space import get_id_model
from nphm.models.deepSDF import GlobalFieldNew
from nphm.models.deformations import DeformationNetwork
from nphm.models.base import LatentCodes







def set_up_networks(cfg,
                    model_type,
                    rank=None,
                    anchors = None,
                    load_dict=None,
                    **kwargs,
                    ):

    if model_type == 'nphm':

        id_decoder = get_id_model(cfg['decoder'],
                                  3 + cfg['decoder']['ex']['nhyper'],
                                  include_color_branch=True,
                                  rank=rank,
                                  )
        ex_decoder = DeformationNetwork(mode=cfg['decoder']['ex']['mode'],
                                            lat_dim_expr=cfg['decoder']['ex']['lat_dim_ex'],
                                            lat_dim_id=cfg['decoder']['ex']['lat_dim_id'],
                                            lat_dim_glob_shape=cfg['decoder']['id']['lat_dim_glob'],
                                            lat_dim_loc_shape=cfg['decoder']['id']['lat_dim_loc_geo'],
                                            n_loc=cfg['decoder']['id']['nloc'],
                                            anchors=anchors,
                                            hidden_dim=cfg['decoder']['ex']['hidden_dim'],
                                            nlayers=cfg['decoder']['ex']['nlayers'],
                                            out_dim=3,
                                            input_dim=3,
                                            neutral_only=False,
                                            n_hyper=cfg['decoder']['ex']['nhyper'],
                                            sdf_corrective=False,  # TODO lambda_sdf_corrective > 0,
                                            local_arch=False,  # TODOlocal_def_arch,
                                            )


    elif model_type == 'global':

        id_decoder = GlobalFieldNew(
        lat_dim=cfg['decoder']['id']['lat_dim'],
        lat_dim_app=cfg['decoder']['id']['lat_dim_app'],
        hidden_dim=cfg['decoder']['id']['hidden_dim'],
        nlayers=cfg['decoder']['id']['nlayers'],
        nlayers_color=cfg['decoder']['id'].get('nlayers_color', 6),
        out_dim=1,
        input_dim=3 + cfg['decoder']['ex']['nhyper'],
        color_branch=True,
        num_freq_bands=cfg['decoder']['id'].get('nfreq_bands_geo', 0),
        freq_exp_base=cfg['decoder']['id'].get('freq_base_geo', 0.5),
        lat_dim_exp=cfg['decoder']['ex']['lat_dim_ex'],
        num_freq_bands_color=cfg['decoder']['id'].get('nfreq_bands_color', 0),
        freq_exp_base_color=cfg['decoder']['id'].get('freq_base_color', 2.0),
        is_monolith=False,
        communication_dim=0,
        uv_communication=False,
            include_anchors=True,
            anchors=anchors,
        )
        ex_decoder = DeformationNetwork(mode=cfg['decoder']['ex']['mode'],
                                            lat_dim_expr=cfg['decoder']['ex']['lat_dim_ex'],
                                            lat_dim_id=-1,
                                            lat_dim_glob_shape=cfg['decoder']['id']['lat_dim'],
                                            lat_dim_loc_shape=-1,
                                            n_loc=-1,  # CFG['decoder']['id']['nloc'],
                                            anchors=None,
                                            hidden_dim=cfg['decoder']['ex']['hidden_dim'],
                                            nlayers=cfg['decoder']['ex']['nlayers'],
                                            out_dim=3,
                                            input_dim=3,
                                            neutral_only=False,
                                            n_hyper=cfg['decoder']['ex']['nhyper'],
                                            sdf_corrective=False,  # TODO lambda_sdf_corrective > 0,
                                            local_arch=False,  # TODOlocal_def_arch,
                                            )
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
    if load_dict:
        id_decoder.load_state_dict(load_dict['id_decoder'])
        ex_decoder.load_state_dict(load_dict['ex_decoder'])

    return id_decoder, ex_decoder
