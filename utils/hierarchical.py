
import logging
logger = logging.getLogger(__name__)
from . import file_tool
import torch
from config import configurator
from argument.tfrs import TFRsPerformingArguments,TFRsDataArguments,TFRsModelArguments
from data.proxy.tfrs import TFRsDataProxy
from framework.transformers.run_glue import TFRsFrameworkProxy
from contrl import controller
from matplotlib import pyplot as plt


class Hierarchic:
    model_path = 'utils/Hierarchic/'

    def __init__(self):
        self.coherence_controller: controller.Controller
        self.coherence_model_path = file_tool.connect_path(self.model_path, 'coherence')

        self.resemblance_controller: controller.Controller
        self.resemblance_model_path = file_tool.connect_path(self.model_path, 'resemblance')
        self._create_controllers()

    def _test_models(self):
        self._load_models(self.coherence_controller, self.coherence_model_path)
        eval_results = self.coherence_controller.framework_proxy.predict()
        # confusion_matrix = eval_results['test_confusion_matrix']
        # plt.matshow(confusion_matrix, cmap=plt.get_cmap('binary'))
        # plt.show()

        # self._load_models(self.resemblance_controller, self.resemblance_model_path)
        # eval_results = self.resemblance_controller.framework_proxy.predict()
        # confusion_matrix = eval_results['test_confusion_matrix']
        # plt.matshow(confusion_matrix, cmap=plt.get_cmap('binary'))
        # plt.show()

    def _create_controllers(self):
        configurator._return_remaining_strings = True

        configurator.register_arguments(model_args=TFRsModelArguments, data_args=TFRsDataArguments, performing_args=TFRsPerformingArguments)
        configurator.data_proxy_type = TFRsDataProxy
        configurator.framework_proxy_type = TFRsFrameworkProxy

        self.coherence_controller = controller.Controller(modify_args={'task_name': 'coherence'})

        self.resemblance_controller = controller.Controller(modify_args={'task_name': 'resemblance'})

    def _load_models(self, controller, model_path_for_load):
        controller.load_pretrained_framework(model_path_for_load)

    def _pretrain_model(self, controller, model_path_for_save):
        _, perform_result = controller.run()
        controller.save_framework(model_path_for_save)

    def _predict_elabration(self):
        mrpc_controller = controller.Controller(modify_args={'task_name': 'mrpc'})
        self.coherence_controller.framework_proxy.predict()


# if __name__ == "__main__":
#







