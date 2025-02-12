import logging
import os

import custom_graphgym  # noqa, register custom modules
import torch

from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg,
    dump_cfg,
    load_cfg,
    set_out_dir,
    set_run_dir,
)
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.model_builder import create_model, create_mlp_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)
    set_out_dir(cfg.out_dir, args.cfg_file)  # 创建输出文件夹
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)  # 将配置信息写入到输出文件夹的配置文件中
    # Repeat for different random seeds
    for i in range(args.repeat):
        set_run_dir(cfg.out_dir)  # 创建当前随机种子的运行文件夹（0）
        set_printing()  # 设置运行日志的输出设备
        # Set configurations for each run
        cfg.seed = cfg.seed + 1
        seed_everything(cfg.seed)  # 设置随机种子
        auto_select_device()  # 选择计算设备
        # Set machine learning pipeline
        datamodule = GraphGymDataModule()  # 加载数据
        if cfg.model.type == 'mlp':
            model = create_mlp_model()  # 创建MLP模型
        else:
            model = create_model()  # 创建GNN模型
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        train(model, datamodule, logger=True)  # 训练和测试

    # Aggregate results from different seeds
    agg_runs(cfg.out_dir, cfg.metric_best)  # precision
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
