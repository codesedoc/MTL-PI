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
    def __init__(self, args=None, study_path=None, study_name=None):
        super().__init__()
        self.args = args
        # self.start_up_trials = 5
        if args!=None:
            self.study_path = file_tool.connect_path("result", self.args.framework_name, 'optuna')
            file_tool.makedir(self.study_path)
            self.study = optuna.create_study(study_name=self.args.framework_name,
                                             storage='sqlite:///' + file_tool.connect_path(self.study_path, 'study_hyper_parameter.db'),
                                             load_if_exists=True,
                                             pruner=optuna.pruners.MedianPruner())
            logger_filename = file_tool.connect_path(self.study_path, 'log.txt')
        else:
            self.study_path = study_path
            self.study = optuna.create_study(study_name=study_name,
                                             storage='sqlite:///' + file_tool.connect_path(study_path,
                                                                                           'study_hyper_parameter.db'),
                                             load_if_exists=True,
                                             pruner=optuna.pruners.MedianPruner())
            logger_filename = file_tool.connect_path(self.study_path, 'log_analysis.txt')
        self.logger = log_tool.get_logger('my_optuna', logger_filename,
                                          log_format=logging.Formatter("%(asctime)s - %(message)s",
                                                                       datefmt="%Y-%m-%d %H:%M:%S"))
        # n_startup_trials = self.start_up_trials, n_warmup_steps = 10

        # self.learn_rate_list = [5e-5, 3e-5, 2e-5, 1e-5]
        self.learn_rate_list = [round(j * math.pow(10, -i), 7) for j in [2, 4, 6, 8] for i in range(4, 7)]
        self.batch_size_list = [16, 32]
        self.transformer_dropout_list = [0, 0.05, 0.1]
        self.gcn_dropout_list = [0, 0.1, 0.2, 0.4]
        self.weight_decay_list = [4 * math.pow(10, -i) for i in range(3, 8, 2)]

        self.trial_times = 500

        if 'trial_dict' in self.study.user_attrs:
            self.trial_dict = self.study.user_attrs['trial_dict']
        else:
            self.trial_dict = {}

    def objective(self, trial):
        self.args.learning_rate = self.learn_rate_list[trial.suggest_int('learn_rate_index', 0, len(self.learn_rate_list)-1)]
        trial.set_user_attr('learning_rate', self.args.learning_rate)

        # self.args.per_gpu_train_batch_size = 8
        self.args.per_gpu_train_batch_size = self.batch_size_list[trial.suggest_int('batch_size_index', 0, len(self.batch_size_list)-1)]
        trial.set_user_attr('batch_size', self.args.per_gpu_train_batch_size)

        self.args.per_gpu_eval_batch_size = self.args.per_gpu_train_batch_size

        self.args.num_train_epochs = trial.suggest_int('epoch', 4, 6)
        trial.set_user_attr('epoch', self.args.num_train_epochs)

        self.args.transformer_dropout = self.transformer_dropout_list[
            trial.suggest_int('transformer_dropout_index', 0, len(self.transformer_dropout_list) - 1)]
        trial.set_user_attr('transformer_dropout', self.args.transformer_dropout)

        self.args.weight_decay = self.weight_decay_list[
            trial.suggest_int('weight_decay_index', 0, len(self.weight_decay_list) - 1)]
        trial.set_user_attr('weight_decay', self.args.weight_decay)

        if self.args.framework_name in self.args.framework_with_gcn:
            self.args.gcn_layer = trial.suggest_int('gcn_hidden_layer', 2, 6)
            trial.set_user_attr('gcn_hidden_layer', self.args.gcn_layer)

            self.args.gcn_dropout = self.gcn_dropout_list[
                trial.suggest_int('gcn_dropout_index', 0, len(self.gcn_dropout_list) - 1)]
            trial.set_user_attr('gcn_dropout', self.args.gcn_dropout)

        self.args.base_learning_rate = 2e-5

        # self.args.start_up_trials = self.start_up_trials
        if trial.number > 0:
            self.log_trial(self.study.best_trial, 'best trial info')

        if str(trial.params) in self.trial_dict:
            self.logger.warning('trail params: %s  repeat!' %(str(trial.params)))
            return self.trial_dict[str(trial.params)]

        framework_manager = fr.FrameworkManager(args=self.args, trial=trial)
        result, attr = framework_manager.run()
        trial.set_user_attr('results', attr)
        torch.cuda.empty_cache()
        self.log_trial(trial, 'current trial info')
        self.trial_dict[str(trial.params)] = result
        return result

    def log_trial(self, trial, head=None):
        self.logger.info('*'*80)
        if head is not None:
            self.logger.info(str(head))

        self.logger.info('number:{}'.format(trial.number))
        self.logger.info('user_attrs:{}'.format(trial.user_attrs))
        self.logger.info('params:{}'.format(trial.params))
        if hasattr(trial, 'state'):
            self.logger.info('state:{}'.format(trial.state))
        self.logger.info('*'*80+'\n')

    def show_best_trial(self):
        # print(dict(self.study.best_trial.params))
        self.log_trial(self.study.best_trial, 'best trial info')

    # def get_real_paras_values_of_trial(self, trial):

    def tune_hyper_parameter(self):
        self.study.optimize(self.objective, n_trials=self.trial_times)
        self.log_trial(self.study.best_trial, 'best trial info')
        self.study.set_user_attr('learn_rate_list', self.learn_rate_list)
        self.study.set_user_attr('batch_size_list', self.batch_size_list)
        file_tool.save_data_pickle(self.study, file_tool.connect_path(self.study_path, 'study_hyper_parameter.pkls'))
        # log_tool.model_result_logger.info(
        #     'Current best value is {} with parameters: {}.'.format(study.best_value, study.best_params))
