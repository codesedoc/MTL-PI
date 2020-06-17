from argument import ModelArguments, PerformingArguments
from typing import Optional, Tuple, Any, Dict, NamedTuple, List
import torch
import numpy as np
from torch.optim.optimizer import Optimizer
import utils.file_tool as file_tool
from utils.general_tool import setup_seed
from data.proxy import DataProxy

import logging
logger = logging.getLogger(__name__)


class PredictionOutput(NamedTuple):
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    indexes: Optional[np.ndarray]
    example_id2pred: Dict[int, np.ndarray] = None
    example_ids: List[int] = None



class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float


class Framework(torch.nn.Module):
    parameter_name_for_no_decay = ["bias"]

    def __init__(self, model_args: ModelArguments, *args, **kwargs):
        super().__init__()
        self.model_args = model_args
        pass

    def get_optimizer_grouped_parameters(self):
        no_decay = self.parameter_name_for_no_decay
        weight_decay = self.model_args.weight_decay
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters
    pass


class FrameworkProxy:
    def __init__(self, model_args: ModelArguments, performing_args: PerformingArguments, data_proxy: DataProxy,
                 *args, **kwargs):
        self.model_args = model_args
        self.performing_args = performing_args
        self.data_proxy = data_proxy
        # setup_seed(model_args.seed)

        self.framework = self._create_framework()
        self.framework.to(self.performing_args.device)

        setup_seed(self.model_args.seed)

        pass

    def _create_framework(self) -> Framework:
        raise NotImplementedError()

    def _get_optimizers(self, num_total_training_steps: int) -> Any:
        raise NotImplementedError()

    def perform(self,  *args, **kwargs):
        performing_args = self.performing_args

        # Training
        if performing_args.do_train:
            logger.info("*** Train ***")
            self.train()
            # self.save_model()

        # Evaluation
        eval_results = {}
        if performing_args.do_eval:
            logger.info("*** Evaluate ***")
            eval_results = self.evaluate()

        # Prediction
        if performing_args.do_predict:
            logging.info("*** Test ***")
            self.predict()

        return eval_results

    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def evaluate(self) -> Dict[str, float]:
        raise NotImplementedError()

    def save_model(self, output_path: Optional[str] = None):
        self.framework.cpu()
        output_path = output_path if output_path is not None else self.performing_args.output_dir
        file_tool.makedir(output_path)

        logger.info("Saving model checkpoint to %s", output_path)

        self.framework.cpu()
        torch.save(self.framework.state_dict(), file_tool.connect_path(output_path, 'framework.pt'))
        self.framework.to(self.performing_args.device)

    def predict(self) -> PredictionOutput:
        raise NotImplementedError()

    def _save_prediction(self, indexes: np.ndarray, predictions: np.ndarray):

        performing_args = self.performing_args
        data_args = self.data_proxy.data_args
        output_mode = self.data_proxy.corpus.output_mode
        from data import OutputMode

        # if output_mode == OutputMode.classification:
        #     predictions = np.argmax(predictions, axis=1)

        if indexes.shape[0] != predictions.shape[0]:
            raise RuntimeError

        output_test_file = file_tool.connect_path(performing_args.output_dir, f"test_results_{data_args.task_name}.txt")
        logger.info("***** Test results {} *****".format(data_args.task_name))

        save_data = ["index\tprediction"]
        for example_index, item in zip(indexes, predictions):
            if output_mode == OutputMode.regression:
                save_data.append("%d\t%3.3f" % (example_index, item))
            else:
                # item = self.data_proxy.corpus.get_labels()[item]
                save_data.append("%d\t%s" % (example_index, item))

        file_tool.save_list_data(save_data, output_test_file, 'w')
        logger.info(f"Save the test results at: {output_test_file}")

        # with open(output_test_file, "w") as writer:
        #
        #     writer.write("index\tprediction\n")
        #
        #     for index, item in enumerate(predictions):
        #         if output_mode == OutputMode.regression:
        #             writer.write("%d\t%3.3f\n" % (index, item))
        #         else:
        #             item = self.data_proxy.corpus.get_labels()[item]
        #             writer.write("%d\t%s\n" % (index, item))

def _create_framework_proxy(proxy_type: type, model_args: ModelArguments, performing_args: PerformingArguments,
                            *args, **kwargs) -> FrameworkProxy:

    framework_proxy = proxy_type(model_args, performing_args, *args, **kwargs)
    return framework_proxy


framework_proxy_singleton: Optional[FrameworkProxy] = None


def create_framework_proxy(proxy_type: type, model_args: ModelArguments, performing_args: PerformingArguments,
                           *args, **kwargs) -> FrameworkProxy:

    global framework_proxy_singleton
    if framework_proxy_singleton is None:
        framework_proxy_singleton = _create_framework_proxy(proxy_type, model_args, performing_args, *args, **kwargs)
    return framework_proxy_singleton
