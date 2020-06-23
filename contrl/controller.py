from config import configurator, Arguments
from data.proxy import create_data_proxy
from framework import create_framework_proxy, FrameworkProxy
import logging
from utils.general_tool import setup_seed
logger = logging.getLogger(__name__)
from optuna import Trial
from dataclasses import replace
import utils.file_tool as file_tool
from socket import gethostname
from typing import Optional
from argument import ArgumentsType
import math


class Controller:
    def __init__(self, trail=None, modify_args=None):
        self.arguments_box = configurator.get_arguments_box()
        self.trail = None
        if trail is not None:
            self.trail = self.load_optuna_trail(trail)
        self._set_base_logger()
        logger.warning(
            "Device: %s, n_gpu: %s, 16-bits training: %s",
            self.arguments_box.performing_args.device,
            self.arguments_box.performing_args.n_gpu,
            self.arguments_box.performing_args.fp16,
        )
        self.data_proxy = None
        self.framework_proxy: Optional[FrameworkProxy] = None
        if modify_args is not None:
            if not isinstance(modify_args, dict):
                raise ValueError
            self.modify_argument(**modify_args)

        self.create_components()
        # self.data_proxy.mute = True
        pass

    def create_components(self):
        logger.info("Arguments: %s", self.arguments_box)
        setup_seed(self.arguments_box.model_args.seed)

        self.data_proxy = create_data_proxy(configurator.data_proxy_type, self.arguments_box.data_args, force=True)

        self.framework_proxy = create_framework_proxy(configurator.framework_proxy_type, self.arguments_box.model_args,
                                                      self.arguments_box.performing_args, self.data_proxy, force=True)

    from typing import Dict, Any

    def _replace_all_arguments_by_dict(self, replace_dict: Dict[str, Any]):
        self.arguments_box.model_args, replace_dict = self.arguments_box.model_args.replace_args(replace_dict)
        self.arguments_box.data_args, replace_dict = self.arguments_box.data_args.replace_args(replace_dict)
        performing_args, replace_dict = self.arguments_box.performing_args.replace_args(replace_dict)

        if len(replace_dict) != 0:
            raise ValueError

        name2abbreviation = performing_args.get_name_abbreviation()
        name2abbreviation.update(self.arguments_box.data_args.get_name_abbreviation())

        #update_tensorboard_path
        path = []
        for hy_name, value in replace_dict.items():

            if hy_name not in name2abbreviation:
                raise ValueError
            path.append(f'{name2abbreviation[hy_name]}-{round(value,8)}')
        if len(path) <=0 :
            if self.trail is not None:
                raise ValueError
            else:
                return
        path = '_'.join(path)
        path = file_tool.connect_path('result/tensorboard', configurator.framework_proxy_type.framework_class.name,  gethostname(),  path)
        self.arguments_box.performing_args = replace(performing_args, logging_dir=path)

    def load_optuna_trail(self, trial: Trial):
        # self.learn_rate_list = [5e-5, 3e-5, 2e-5, 1e-5]
        # self.learn_rate_list = [round(j * math.pow(10, -i), 7) for j in [2, 4, 6, 8] for i in range(4, 7)]
        # batch_size_list = [16, 32]
        # self.transformer_dropout_list = [0, 0.05, 0.1]

        # self.weight_decay_list = [4 * math.pow(10, -i) for i in range(3, 8, 2)]
        real_hyps = {}

        learning_rate = trial.suggest_loguniform('learning_rate', 8e-6, 8e-4)
        real_hyps['learning_rate'] = learning_rate

        # per_device_train_batch_size = batch_size_list[trial.suggest_int('batch_size', 0, len(batch_size_list)-1)]
        # real_hyps['per_device_train_batch_size'] = per_device_train_batch_size
        #
        num_train_epochs = trial.suggest_int('epoch', 3, 5)
        real_hyps['num_train_epochs'] = num_train_epochs
        #
        auxiliary_learning_rate = trial.suggest_loguniform('auxiliary_learning_rate', 2e-6, 2e-4)
        real_hyps['auxiliary_learning_rate'] = auxiliary_learning_rate
        #
        # auxiliary_per_device_batch_size = batch_size_list[trial.suggest_int('auxiliary_batch_size', 0, len(batch_size_list)-1)]
        # real_hyps['auxiliary_per_device_batch_size'] = auxiliary_per_device_batch_size
        #

        # import torch
        # print(torch.randn(10))

        auxiliary_training_epoch = trial.suggest_int('auxiliary_training_epoch', 6, 7)
        real_hyps['auxiliary_training_epoch'] = auxiliary_training_epoch
        #

        trial.set_user_attr('real_hyper_params', real_hyps)

        self._replace_all_arguments_by_dict(real_hyps)

        return trial

    def _set_base_logger(self):
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

    def run(self):
        perform_result = self.framework_proxy.perform(self.data_proxy)
        eval_results = perform_result['eval_results']

        standard = 1 - eval_results.get('dev_acc')

        if self.framework_proxy.tb_writer is not None and self.trail is not None:
            from utils.general_tool import is_number
            hp_metric_dict = {f'hparams/{k}': v for k, v in eval_results.items() if is_number(str(v))}

            self.framework_proxy.tb_writer.add_hparams(self.trail.user_attrs['real_hyper_params'], metric_dict=hp_metric_dict)

        return standard, perform_result

    def modify_argument(self, **name2value):
        self._replace_all_arguments_by_dict(name2value)

    def load_pretrained_framework(self, model_path):
        self.framework_proxy.load_model(model_path)

    def save_framework(self, model_path):
        self.framework_proxy.save_model(model_path)


