#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._perm_path = ""
        self._obj_head_path = ""
        self._loaded_roots_path = ""
        self._nphm_config_path = ""
        self._geo_nphm_path = ""
        self._app_nphm_path = ""
        self._xp_nphm_path = ""
        self._dict_loaded_nphm_path = ""
        self._cached_roots_path = "roots.pt"
        self._emp_hair_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self._kernel_size = 0.1
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 150_000
        self.theta_warmup = 1

        self.theta_lr_init = 0.000250
        self.theta_lr_final = 0.00025
        self.beta_lr_init = 0.000250
        self.beta_lr_final = 0.00025
        self.perm_lr_delay_mult = 0.01
        self.perm_lr_max_steps = 300_000

        # self.position_lr_init = 0.00016
        # self.position_lr_final = 0.0000016
        # self.position_lr_delay_mult = 0.01
        # self.position_lr_max_steps = 30_000

        lr_coef = 1
        self.feature_lr = 0.0075*lr_coef
        self.opacity_lr = 0.00015*lr_coef
        self.scaling_lr = 0.001*lr_coef
        self.rotation_lr = 0.005*lr_coef
        self.percent_dense = 1e-4

        self.lambda_neighbor_orient = 1.0
        self.max_strand_len = 0.22
        self.delta_strand_len = 0.01
        self.k_neigh = 8

        self.lambda_huber = 400.0
        self.lambda_seg = 10000.0 * 5.0
        self.lambda_orient = 125.0 * 10.0
        self.lambda_sdf_contain = 0.0
        self.lambda_sdf_flow = 3e3
        self.lambda_neigh = 0.1
        self.lambda_out = 0.0
        self.lambda_ori_match = 1e2
        self.lambda_oblong = 1e9
        self.lambda_len_consist = 1e11
        self.lambda_bend = 5e8
        self.lambda_smooth_scale = 1e7
        self.lambda_sobel = 1e27
        self.lambda_head_col = 30000.0
        self.lambda_gauss_head_col = 30000.0
        self.lambda_local_len = 300000.0
        self.lambda_color_var = 3000.0
        self.lambda_opacity_var = 1e8
        self.lambda_theta_l2 = 0.000000 # 4
        self.lambda_beta_l2 = 0.000000
        self.lambda_scale_reg = 0.0

        self.densification_interval = 100 # 7500
        self.opacity_reset_interval = 3500
        self.densify_from_iter = 99
        self.densify_until_iter = 99 # 22500
        self.densify_grad_threshold = 0.0002

        self.densify_strands_from_iter = 100
        self.densify_strands_until_iter = 100 # 25000
        self.densification_strand_interval = 2000 # 5000
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

