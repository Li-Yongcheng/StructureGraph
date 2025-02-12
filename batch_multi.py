import os

from torch_geometric.graphgym.cmd_args import parse_args
from concurrent.futures import ProcessPoolExecutor


def run_experiment(cfg_file_path, repeat):
    """根据配置文件运行1个实验, 使用随机种子跑n次"""
    os.system(f'python main.py --cfg {cfg_file_path} --repeat {repeat} --mark_done')


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()

    future_results = []
    with ProcessPoolExecutor(max_workers=15) as pool:
        for cfg_file in os.listdir(args.cfg_file):
            if cfg_file.split('.')[-1] != 'yaml':  # 检查是否为配置文件
                continue
            cfg_file_path = os.path.join(args.cfg_file, cfg_file)

            future_result = pool.submit(run_experiment, cfg_file_path, args.repeat)
            future_results.append(future_result)
