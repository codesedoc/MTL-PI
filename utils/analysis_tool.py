import math
import utils.file_tool as file_tool
import framework as fr


class HyperParameterAnalyst:
    def __init__(self, args):
        self.args = args
        self.max_epoch = 15
        self.set_hyper_parameters()
        pass

    def set_hyper_parameters(self):
        self.args.transformer_dropout = 0.1
        self.args.gcn_dropout = 0.2
        self.args.without_concatenate_input_for_gcn_hidden = False
        self.args.gcn_layer = 4
        self.args.do_train = True
        self.args.do_eval = True
        self.args.do_test = True
        self.args.evaluate_during_training = True
        self.args.per_gpu_train_batch_size = 8
        self.args.per_gpu_eval_batch_size = 8
        self.args.learning_rate = 0.0005
        self.args.weight_decay = 0.0
        self.args.num_train_epochs = self.max_epoch
        self.args.logging_steps = 100
        self.args.base_learning_rate = 2e-05
        self.args.gcn_gate_flag = True
        self.args.gcn_norm_item = 0.5
        self.args.gcn_self_loop_flag = True
        self.args.gcn_group_layer_limit_flag = False
        self.args.gcn_position_encoding_flag = True

    def analyze_learning_rate(self):
        lrs = [j * math.pow(10, -i)for j in [2, 5, 8] for i in range(1, 8)]
        # lrs = [0.005]
        base_path = 'tensorboard/tuning_test/learning_rate'
        for lr in lrs:
            self.args.learning_rate = lr
            self.args.tensorboard_logdir = file_tool.connect_path(base_path, str(lr))
            framework_manager = fr.FrameworkManager(self.args)
            framework_manager.run()
