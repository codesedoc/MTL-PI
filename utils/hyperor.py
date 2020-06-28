import optuna
import utils.file_tool as file_tool
import math
import utils.general_tool as general_tool
import torch
import framework as fr
import utils.log_tool as log_tool
import logging



# class HyperParameter:
#     def __init__(self, short_name, args_pointer, value):
#         self.short_name = short_name
#         self.args_pointer = short_name

class Hyperor:
    def __init__(self, study_path, study_name, trial_times=10):
        super().__init__()
        # self.args = args
        # # self.start_up_trials = 5
        # if args!=None:
        #     self.study_path = file_tool.connect_path("result", self.args.framework_name, 'optuna')
        #     file_tool.makedir(self.study_path)
        #     self.study = optuna.create_study(study_name=self.args.framework_name,
        #                                      storage='sqlite:///' + file_tool.connect_path(self.study_path, 'study_hyper_parameter.db'),
        #                                      load_if_exists=True,
        #                                      pruner=optuna.pruners.MedianPruner())
        #     logger_filename = file_tool.connect_path(self.study_path, 'log.txt')
        # else:
        if study_path is None:
            study_path = 'result/optuna'

        import os
        gpu_num = os.environ['CUDA_VISIBLE_DEVICES']

        if len(gpu_num) != 1:
            raise ValueError

        study_path = file_tool.connect_path(study_path, f'gpu_{gpu_num}')

        self.study_path = study_path
        if not file_tool.exists(study_path):
            file_tool.makedir(study_path)

        self.study = optuna.create_study(study_name=study_name,
                                         storage='sqlite:///' + file_tool.connect_path(study_path,
                                                                                       'study_hyper_parameter.db'),
                                         load_if_exists=True,
                                         pruner=optuna.pruners.MedianPruner())

        logger_filename = file_tool.connect_path(self.study_path, 'log_analysis.txts')
        self.logger = log_tool.get_logger('my_optuna', logger_filename,
                                          log_format=logging.Formatter("%(asctime)s - %(message)s",
                                                                       datefmt="%Y-%m-%d %H:%M:%S"))

        self.trial_times = trial_times

        if 'trial_dict' in self.study.user_attrs:
            self.trial_dict = self.study.user_attrs['trial_dict']
        else:
            self.trial_dict = {}
            self.study.set_user_attr('trial_dict', self.trial_dict)

    def objective(self, trial):

        from contrl.controller import Controller

        controller = Controller(trial)
        controller.add_user_atts_to_trial(trial)

        trial = controller.trail

        hyper_params = trial.user_attrs['real_hyper_params']

        if str(hyper_params) in self.trial_dict:
            self.logger.info('*'*80)
            self.logger.info('*************Repeat!**************\n')
            self.logger.info('trail hyper_params: %s  repeat!' % (str(hyper_params)))
            self.logger.info(f'corresponding result: {self.trial_dict[str(hyper_params)]}')

            best_trial = self.study.best_trial
            self.logger.info(f'best trial number:{best_trial.number} and result:{best_trial.user_attrs["result"]}')
            self.logger.info('*'*80+'\n')

            return self.trial_dict[str(hyper_params)]

        result, attr = controller.run()

        del controller

        torch.cuda.empty_cache()

        if not general_tool.is_number(result):
            raise ValueError

        self.record_one_time_trial(trial, result, attr, hyper_params)

        return result

    def record_one_time_trial(self, trial: optuna.Trial, result, attr, hyper_params):

        if isinstance(attr, dict):
            for k, v in attr.items():
                trial.set_user_attr(k, v)
        else:
            trial.set_user_attr('other_results', attr)
        trial.set_user_attr('result', result)

        tail = None
        if trial.number > 0:
            best_trial = self.study.best_trial
            tail = f'best trial number:{best_trial.number} and result:{best_trial.user_attrs["result"]}'
        self.log_trial(trial, 'Current Trial Info', tail=tail)

        self.trial_dict[str(hyper_params)] = result
        self.study.set_user_attr('trial_dict', self.trial_dict)

    def log_trial(self, trial, head=None, tail=None):
        self.logger.info('*'*80)
        if head is not None:
            self.logger.info(str(head))

        self.logger.info('number:{}'.format(trial.number))
        self.logger.info('user_attrs:{}'.format(trial.user_attrs))
        self.logger.info('params:{}'.format(trial.params))
        if hasattr(trial, 'state'):
            self.logger.info('state:{}'.format(trial.state))

        if tail is not None:
            self.logger.info("")
            self.logger.info(str(tail))

        self.logger.info('*'*80+'\n')

    def show_best_trial(self):
        # print(dict(self.study.best_trial.params))
        self.log_trial(self.study.best_trial, 'Best Trial Info')

    # def get_real_paras_values_of_trial(self, trial):

    def tune_hyper_parameter(self):
        self.study.optimize(self.objective, n_trials=self.trial_times)

        tail = f'{"#"*10} Optuna finish another {self.trial_times} trials! {"#"*10}'
        self.log_trial(self.study.best_trial, 'Best Trial Info', tail=tail)
        file_tool.save_data_pickle(self.study, file_tool.connect_path(self.study_path, 'study_hyper_parameter.pkls'))
        # log_tool.model_result_logger.info(
        #     'Current best value is {} with parameters: {}.'.format(study.best_value, study.best_params))
