import argparse

from config import configurator
from utils.hyperor import Hyperor

from argument.tfrs import TFRsPerformingArguments,TFRsDataArguments,TFRsModelArguments
from data.proxy.tfrs import TFRsDataProxy
from framework.transformers.run_glue import TFRsFrameworkProxy

from argument.mtl_pi import MTLPIPerformingArguments, MTLPIModelArguments, MTLPIDataArguments
from data.proxy.mtl_pi import MTLPIDataProxy
from framework.mtl.mtl_pi import MTLPIFrameworkProxy

configurator._return_remaining_strings = True

# configurator.register_arguments(model_args = TFRsModelArguments, data_args=TFRsDataArguments, performing_args=TFRsPerformingArguments)
# configurator.data_proxy_type = TFRsDataProxy
# configurator.framework_proxy_type = TFRsFrameworkProxy


configurator.register_arguments(model_args=MTLPIModelArguments,
                                data_args=MTLPIDataArguments,
                                performing_args=MTLPIPerformingArguments)

configurator.data_proxy_type = MTLPIDataProxy
configurator.framework_proxy_type = MTLPIFrameworkProxy


parser = argparse.ArgumentParser()
parser.add_argument(
        "-th",
        "--tune_hyper",
        default=False,
        action="store_true",
        help="Whether tune hyper-parameter",
    )

args, _ = parser.parse_known_args()
# tuning_hp = False
tuning_hp = args.tune_hyper

if tuning_hp:
    framework_name = configurator.framework_proxy_type.framework_class.name
    import utils.file_tool as file_tool
    from socket import gethostname
    thp = Hyperor(study_path=file_tool.connect_path('result/optuna', framework_name, gethostname()), study_name=framework_name, trial_times=None)
    thp.tune_hyper_parameter()
else:
    from contrl import controller

    c = controller.Controller()
    c.run()
    c.save_examples_according_to_evaluation()





# import utils.hyperor as hyperor
#
# thp = hyperor.Hyperor(study_path = 'tmp')
# thp.tune_hyper_parameter()

# import optuna
#
# def objective(trial: optuna.Trial):
#     x = trial.suggest_loguniform('x', 1, 10)
#     print(x)
#     return x
#
#
# study = optuna.create_study()
# study.optimize(objective, n_trials=100)

# from dataclasses import fields
# print( [f.name for f in fields(MTLPIDataArguments)])

