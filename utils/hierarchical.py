
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
        from contrl.controller import ComponentEnum
        from data.proxy import DataSetType

        def get_mrpc_elab_preds():

            coherence_predict_result = self.coherence_controller.framework_proxy.predict_other_data(mrpc_data_proxy, ds_type=DataSetType.train)

            coherence_e_id2preds = coherence_predict_result.example_id2pred

            resemblance_predict_result = self.resemblance_controller.framework_proxy.predict_other_data(mrpc_data_proxy, ds_type=DataSetType.train)

            resemblance_e_id2preds = resemblance_predict_result.example_id2pred

            mrpc_e_ids = set(coherence_predict_result.example_id2pred.keys())
            # mrpc_label = [mrpc_data_proxy.corpus.id2example[e_id].label for e_id in mrpc_e_ids]

            # mrpc_data_predict = []

            from data.corpus.discourse.coherence.coherence import Coherence
            from data.corpus.discourse.resemblance.resemblance import Resemblance

            e_yes = set()
            e_no = set()
            for e_id in mrpc_e_ids:
                if coherence_e_id2preds[e_id] == Coherence.resemblance.value and resemblance_e_id2preds[e_id] == Resemblance.elab.value:
                    e_yes.add(e_id)

                else:
                    e_no.add(e_id)

            if len(e_no) + len(e_yes) != len(mrpc_e_ids):
                raise ValueError

            return e_yes, e_no, mrpc_e_ids

        self._load_models(self.coherence_controller, self.coherence_model_path)
        self._load_models(self.resemblance_controller, self.resemblance_model_path)

        self.coherence_controller.modify_argument(task_name='mrpc')
        mrpc_data_proxy = self.coherence_controller.create_component(ComponentEnum.data_proxy)

        mrpc_data_proxy.tokenizer = self.coherence_controller.data_proxy.tokenizer
        mrpc_data_proxy.compute_metrics_function = self.coherence_controller.data_proxy.compute_metrics_function
        mrpc_data_proxy.get_dataloader(DataSetType.train, force=True)

        o_e_y, o_e_n, o_e_ids = get_mrpc_elab_preds()
        print(id(mrpc_data_proxy.corpus.type2examples[DataSetType.train]))
        mrpc_data_proxy.get_dataloader(DataSetType.train, force=True, reverse_texts_order=True)
        print(id(mrpc_data_proxy.corpus.type2examples[DataSetType.train]))

        r_e_y, r_e_n, r_e_ids = get_mrpc_elab_preds()

        for o_e_id in o_e_ids:
            if o_e_id not in r_e_ids:
                raise ValueError

        from data.corpus.discourse.elaboration.elaboration import Elaboration
        from data.corpus.glue.mrpc.mrpc import ParapraseLabel

        e_id2elab_pred = {}
        e_y = []
        e_n = []
        for e_id in o_e_ids:
            if e_id in o_e_y:
                if e_id in r_e_n:
                    e_id2elab_pred[e_id] = Elaboration.yes
                    e_y.append(e_id)
                else:
                    e_id2elab_pred[e_id] = Elaboration.no
                    e_n.append(e_id)
            else:
                if e_id in r_e_y:
                    e_id2elab_pred[e_id] = Elaboration.yes
                    e_y.append(e_id)
                else:
                    e_id2elab_pred[e_id] = Elaboration.no
                    e_n.append(e_id)

        e_y_p_y = []
        e_y_p_n = []
        e_n_p_y = []
        e_n_p_n = []
        for e_id in e_y:
            if mrpc_data_proxy.corpus.id2example[e_id].label == ParapraseLabel.yes:
                e_y_p_y.append(e_id)
            else:
                e_y_p_n.append(e_id)
        for e_id in e_n:
            if mrpc_data_proxy.corpus.id2example[e_id].label == ParapraseLabel.yes:
                e_n_p_y.append(e_id)
            else:
                e_n_p_n.append(e_id)

        mrpc_data_proxy.output_examples(e_y_p_y, 'utils/Hierarchic/predict_elab_yes_and_paraphraze_yes.txt')
        mrpc_data_proxy.output_examples(e_y_p_n, 'utils/Hierarchic/predict_elab_yes_and_paraphraze_no.txt')

        mrpc_data_proxy.output_examples(e_n_p_y, 'utils/Hierarchic/predict_elab_no_and_paraphraze_yes.txt')
        mrpc_data_proxy.output_examples(e_n_p_n, 'utils/Hierarchic/predict_elab_no_and_paraphraze_no.txt')

        file_tool.save_data_pickle(e_id2elab_pred, 'utils/Hierarchic/mrpc_eid2elab_pred.pkl')
        pass



# if __name__ == "__main__":
#







