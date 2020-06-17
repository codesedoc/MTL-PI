import torch
from torch.utils.tensorboard import SummaryWriter
import utils.file_tool as file_tool
import os
import time


def log_graph(filename,  nn_model, input_data, comment =None,):
    with SummaryWriter(filename, comment=comment) as w:
        w.add_graph(nn_model, input_data)


def run_tensorboard_command():
    command = 'tensorboard --logdir='+file_tool.PathManager.tensorboard_runs_path
    os.system(command)


def create_filename(path):

    filename = time.strftime("%Y_%m_%d %H_%M_%S", time.localtime())
    filename = os.path.join(path, filename)

    return filename