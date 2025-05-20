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
        self._emp_hair_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
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
        self.iterations = 600_000
        self.theta_warmup = 2

        self.perm_lr_init = 0.001
        self.perm_lr_final = 0.0005
        self.perm_lr_delay_mult = 0.01
        self.perm_lr_max_steps = 300_000

        # self.position_lr_init = 0.00016
        # self.position_lr_final = 0.0000016
        # self.position_lr_delay_mult = 0.01
        # self.position_lr_max_steps = 30_000

        lr_coef = 1
        self.feature_lr = 0.0025*lr_coef
        self.opacity_lr = 0.01*lr_coef
        self.scaling_lr = 0.0001*lr_coef
        self.rotation_lr = 0.005*lr_coef
        self.percent_dense = 0.01

        self.lambda_neighbor_orient = 1.0
        self.max_strand_len = 0.22
        self.delta_strand_len = 0.01
        self.k_neigh = 8

        self.lambda_huber = 400.0
        self.lambda_seg = 1000.0
        self.lambda_orient = 10.0
        self.lambda_len = 100.0
        self.lambda_neigh = 50.0
        self.lambda_out = 0.0
        self.lambda_ori_match = 1e2
        self.lambda_oblong = 1e9
        self.lambda_len_consist = 1e11
        self.lambda_bend = 5e7
        self.lambda_smooth_scale = 1e8
        self.lambda_depth = 0.0
        self.lambda_head_col = 0.0
        self.lambda_strand_rep = 10.0

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
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

