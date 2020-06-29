from config import configurator, Arguments
from data.proxy import create_data_proxy
from framework import create_framework_proxy, FrameworkProxy
import logging
from utils.general_tool import setup_seed
from optuna import Trial
from dataclasses import replace
import utils.file_tool as file_tool
from socket import gethostname
from typing import Optional
from enum import Enum, unique
from typing import Dict, Any, Optional
from argument import ArgumentsType
import math
logger = logging.getLogger(__name__)


@unique
class ComponentEnum(Enum):
    # arguments_box = 'arguments_box'
    data_proxy = 'data_proxy'
    framework_proxy = 'framework_proxy'


class Controller:
    def __init__(self, trail=None, modify_args=None):
        self._set_base_logger()

        self.arguments_box = configurator.get_arguments_box()
        self.trail = None
        if trail is not None:
            self.trail = self.load_optuna_trail(trail)

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

        # self.modify_argument(n_args=1)

        self.create_components()
        # self.data_proxy.mute = True
        pass

    def create_components(self):
        logger.info("Arguments: %s", self.arguments_box)
        setup_seed(self.arguments_box.model_args.seed)

        self.data_proxy = create_data_proxy(configurator.data_proxy_type, self.arguments_box.data_args, force=True)

        self.framework_proxy = create_framework_proxy(configurator.framework_proxy_type, self.arguments_box.model_args,
                                                      self.arguments_box.performing_args, self.data_proxy, force=True)

    from argument import ArgumentsBox

    def create_component(self, component_name: ComponentEnum, arguments_box: Optional[ArgumentsBox] = None):
        if arguments_box is None:
            arguments_box = self.arguments_box
        if component_name == ComponentEnum.data_proxy:
            result = create_data_proxy(configurator.data_proxy_type, arguments_box.data_args, force=True)
        elif component_name == ComponentEnum.framework_proxy:
            result = create_framework_proxy(configurator.framework_proxy_type, arguments_box.model_args,
                                            arguments_box.performing_args, self.data_proxy, force=True)
        else:
            raise ValueError
        return result

    def _replace_all_arguments_by_dict(self, replace_dict: Dict[str, Any]):
        replace_dict_  = replace_dict.copy()
        self.arguments_box.model_args = self.arguments_box.model_args.replace_args(replace_dict_)
        self.arguments_box.data_args = self.arguments_box.data_args.replace_args(replace_dict_)
        performing_args = self.arguments_box.performing_args.replace_args(replace_dict_)

        if len(replace_dict_) != 0:
            raise ValueError

        name2abbreviation = performing_args.get_name_abbreviation()

        name2abbreviation.update(self.arguments_box.data_args.get_name_abbreviation())
        name2abbreviation.update(self.arguments_box.model_args.get_name_abbreviation())

        #update_tensorboard_path
        path = []
        for hy_name, value in replace_dict.items():

            if hy_name not in name2abbreviation:
                logging.error(f"hy{hy_name} is not in name2abbreviation")
                return
            path.append(f'{name2abbreviation[hy_name]}-{round(value,8)}')
        if len(path) <= 0:
            logger.info("Need not modify tensorboard path when modify some args")
            return
        if self.trail is not None:
            logger.warning("Modify tensorboard path is not because of trial")

        path = '_'.join(path)

        import os
        gpu_num = os.environ['CUDA_VISIBLE_DEVICES']

        if len(gpu_num) != 1:
            raise ValueError

        path = file_tool.connect_path('result/tensorboard', configurator.framework_proxy_type.framework_class.name,  gethostname(), f'gpu_{gpu_num}', path)
        self.arguments_box.performing_args = replace(performing_args, logging_dir=path)

    def load_optuna_trail(self, trial: Trial):
        # self.learn_rate_list = [5e-5, 3e-5, 2e-5, 1e-5]
        # self.learn_rate_list = [round(j * math.pow(10, -i), 7) for j in [2, 4, 6, 8] for i in range(4, 7)]
        batch_size_list = [16, 32]
        # self.transformer_dropout_list = [0, 0.05, 0.1]

        # self.weight_decay_list = [4 * math.pow(10, -i) for i in range(3, 8, 2)]
        real_hyps = {}
        import random

        def _hyps_case(ranges):
            if len(ranges) == 0:
                return [[]]
            result = []
            for value in ranges[0]:
                for cases_of_sub_seq in _hyps_case(ranges[1:]):
                    new_case = [value]
                    new_case.extend(cases_of_sub_seq)
                    result.append(new_case)
            return result

        def _sample_hyps(sample_count):
            real_hyps.clear()

            if sample_count <= my_self_sample_threshold:
                learning_rate = round(trial.suggest_int('learning_rate', 8, 80) * 1e-6, 8)                                ##################
                per_device_train_batch_size = batch_size_list[trial.suggest_int('batch_size', 0, len(batch_size_list)-1)] ########################
                num_train_epochs = trial.suggest_int('epoch', 2, 4)                                                       ###################
            else:
                trial.set_user_attr('info', "create hyparameters by my self")
                learning_rate = round(random.randint(8, 80) * 1e-6, 8)                                                   ##########################
                per_device_train_batch_size = batch_size_list[random.randint(0, len(batch_size_list)-1)]                ##############
                num_train_epochs = random.randint(2, 4)                                                                 ########################

            real_hyps['learning_rate'] = learning_rate
            real_hyps['per_device_train_batch_size'] = per_device_train_batch_size
            real_hyps['num_train_epochs'] = num_train_epochs

            trial.set_user_attr('real_hyper_params', real_hyps.copy())

            if sample_count == sample_limitation:
                key_of_trial = Hyperor.key_of_one_trial(trial)
                if key_of_trial not in keys_of_tried_trial:
                    return
                trial.set_user_attr('info', "create hyparameters by my self")

                hyps_ranges = []
                hyps_ranges.append([round(factor * 1e-6, 8)for factor in range(8, 80+1)])                               ########################
                hyps_ranges.append(batch_size_list)                                                                     #########################
                hyps_ranges.append(list(range(2, 4+1)))                                                                   ########################
                cases = _hyps_case(hyps_ranges)

                logging.info(f"The total case of hyperparameters: {len(cases)}")

                for case in cases:
                    real_hyps['learning_rate'] = case[0]                                                              ####################
                    real_hyps['per_device_train_batch_size'] = case[1]                                                ########################
                    real_hyps['num_train_epochs'] = case[2]                                                             ##################################

                    trial.set_user_attr('real_hyper_params', real_hyps.copy())

                    key_of_trial = Hyperor.key_of_one_trial(trial)
                    if key_of_trial not in keys_of_tried_trial:
                        return

            # auxiliary_learning_rate = round(trial.suggest_int('auxiliary_learning_rate', 1, 5) * 1e-5, 8)
            # real_hyps['auxiliary_learning_rate'] = auxiliary_learning_rate
            #
            # auxiliary_per_device_batch_size = batch_size_list[trial.suggest_int('auxiliary_batch_size', 0, len(batch_size_list)-1)]
            # real_hyps['auxiliary_per_device_batch_size'] = auxiliary_per_device_batch_size
            #

            # import torch
            # print(torch.randn(10))

            # auxiliary_training_epoch = trial.suggest_int('auxiliary_training_epoch', 2, 3)
            # real_hyps['auxiliary_training_epoch'] = auxiliary_training_epoch

            # loss_a = trial.suggest_loguniform('loss_a', 0.01, 1.0)
            # real_hyps['loss_a'] = loss_a

            #

        from utils.hyperor import Hyperor
        keys_of_tried_trial = trial.tried_trial_keys
        sample_count = 0
        my_self_sample_threshold = 50
        sample_limitation = 100
        import time
        strat_time = time.time()
        time_limitation = 3600*0.5
        while True:
            sample_count += 1
            # strat_time = time.time()
            _sample_hyps(sample_count)
            used_tiem = time.time()-strat_time

            # print(f'used tiem: {time.time()-strat_time}')

            key_of_trial = Hyperor.key_of_one_trial(trial)
            if key_of_trial not in keys_of_tried_trial:
                logging.info(f'Finish load the {len(keys_of_tried_trial) +1 }-th trial. '
                             f'After sample {sample_count} times hyperparameters for it'
                             f'\n It is {key_of_trial}')
                break

            if sample_count >= sample_limitation:
                logging.warning(f'After sample {sample_count} time hyperparameters for this trial, out of the sample limitation! '
                                f'Used time:{round(used_tiem, 2)} seconds!  '
                                f'So, stop load this trial! The number of trial tried is {len(keys_of_tried_trial)}')
                raise ValueError ("over sample limitation")

            if used_tiem> time_limitation:
                logging.warning(f'After sample {sample_count} time hyperparameters for this trial, out of the time limitation! '
                                f'Used time:{round(used_tiem, 2)} seconds!  '
                                f'So, stop load this trial! The number of trial tried is {len(keys_of_tried_trial)}')
                raise ValueError ("over time")

        self._replace_all_arguments_by_dict(real_hyps)

        return trial

    def add_user_atts_to_trial(self, trial: Trial):
        trial.set_user_attr('args_need_to_record', self.framework_proxy.args_need_to_record())

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

        if self.framework_proxy.tb_writer is not None:
            if self.trail is not None:
                from utils.general_tool import is_number
                hp_metric_dict = {f'hparams/{k}': v for k, v in eval_results.items() if is_number(str(v))}

                self.framework_proxy.tb_writer.add_hparams(self.trail.user_attrs['real_hyper_params'], metric_dict=hp_metric_dict)

            test_results = perform_result.get('test_results')
            if test_results is not None:
                import json
                self.framework_proxy.tb_writer.add_text("test_results", json.dumps(test_results, indent=2))

        return standard, perform_result

    def modify_argument(self, **name2value):
        self._replace_all_arguments_by_dict(name2value)

    def load_pretrained_framework(self, model_path):
        self.framework_proxy.load_model(model_path)

    def save_framework(self, model_path):
        self.framework_proxy.save_model(model_path)

    def save_examples_according_to_evaluation(self):
        self.data_proxy.save_examples_according_to_evaluation(output_dir=self.arguments_box.performing_args.output_dir)


