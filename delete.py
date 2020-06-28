import os
from utils import file_tool
# rm -r framework/mtl/after_auxiliary/


gpu_num = os.environ['CUDA_VISIBLE_DEVICES']

if len(gpu_num) != 1:
    raise ValueError

from socket import gethostname

tensorboard_path = f'result/tensorboard/MTL_PI/{gethostname()}/gpu_{gpu_num}'

optuna_path = f'result/optuna/MTL_PI/{gethostname()}/gpu_{gpu_num}'

auxiliary_path = 'framework/mtl/after_auxiliary/'

file_path = [auxiliary_path, tensorboard_path, optuna_path]

for fp in file_path:
    if file_tool.check_dir(fp):
        str = os.popen(f"rm -r {fp}").read()
        print(str)
    else:
        print(f'{fp} do not exist!')

# a = str.split("\n")
# for b in a:
#     print(b)