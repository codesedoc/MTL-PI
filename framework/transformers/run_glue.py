from .. import Framework, FrameworkProxy
from argument.tfrs import TFRsModelArguments, TFRsPerformingArguments
from data.proxy.tfrs import TFRsDataProxy
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
# from .._utils import AdamW, get_linear_schedule_with_warmup
from typing import Any, Tuple, Dict, Optional, List, Union
from dataclasses import dataclass, replace
from data.proxy import DataSetType
from torch.utils.tensorboard import SummaryWriter
import torch
import utils.file_tool as file_tool
from .. import OptimizationKit

import logging
logger = logging.getLogger(__name__)


class TFRsFramework(Framework):
    name = 'TFRs'
    def __init__(self, model_args: TFRsModelArguments, *args, **kwargs):
        super().__init__(model_args, *args, **kwargs)
        config = kwargs['config']

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    def forward(self, **input_):
        tfrs_input = {}
        for name in ['labels', 'input_ids', 'attention_mask', 'token_type_ids']:
            if name in input_:
                tfrs_input[name] = input_[name]
        return self.model(**tfrs_input)


class TFRsFrameworkProxy(FrameworkProxy):
    framework_class = TFRsFramework

    def __init__(self, model_args: TFRsModelArguments, performing_args: TFRsPerformingArguments, data_proxy: TFRsDataProxy,
                 *args, **kwargs):
        self.data_proxy = data_proxy

        self.data_proxy.tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

        super().__init__(model_args, performing_args, data_proxy, *args, **kwargs)


        # self.data_proxy.get_dataset(DataSetType.train)
        # self.data_proxy.get_dataset(DataSetType.dev)

    def _create_framework(self):
        model_args = self.model_args

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=self.data_proxy.num_label,
            finetuning_task=self.data_proxy.task_name,
            cache_dir=model_args.cache_dir,
        )

        return TFRsFramework(model_args, config=config)

        pass

    def _get_optimization_kit(self, num_total_training_steps, force: bool = False, **kwargs) -> OptimizationKit:
        if self.optimization_kit is None or force:
            optimizer_grouped_parameters = self.framework.get_optimizer_grouped_parameters()
            args = self.performing_args
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_total_training_steps
            )
            self.optimization_kit = OptimizationKit(optimizer, scheduler)

        return self.optimization_kit

    def _calculate_loss(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = model(**inputs)
        loss = outputs[0]
        return loss

    def evaluate(self):
        args = self.performing_args
        data_args = self.data_proxy.data_args
        eval_results = {}

        def do_evaluate():

            eval_result = self._evaluate(DataSetType.dev)

            file_tool.makedir(args.output_dir)

            output_eval_file = file_tool.connect_path(args.output_dir, f"eval_results_{data_args.task_name}.txt")

            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval dev set results {} *****".format(data_args.task_name))
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
            eval_results.update(eval_result)

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        do_evaluate()
        if data_args.task_name == "mnli":
            replace(data_args, task_name="mnli-mm")
            do_evaluate()
            replace(data_args, task_name="mnli")

        return eval_results

    def predict(self):
        output_mode = self.data_proxy.corpus.output_mode
        args = self.performing_args
        data_args = self.data_proxy.data_args
        eval_results = {}
        def do_predict():
            predict_output = self._predict()

            if (predict_output.metrics is not None) and len(predict_output.metrics) > 0 and \
                    (predict_output.label_ids is not None):
                metrics_result = predict_output.metrics

                file_tool.makedir(args.output_dir)
                output_eval_file = file_tool.connect_path(args.output_dir, f"eval_results_{data_args.task_name}.txt")

                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval test set results {} *****".format(data_args.task_name))
                    for key, value in metrics_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                eval_results.update(metrics_result)

            self._save_prediction(indexes=predict_output.indexes, predictions=predict_output.predictions)

        do_predict()
        if data_args.task_name == "mnli":
            replace(data_args, task_name="mnli-mm")
            do_predict()
            replace(data_args, task_name="mnli")

        if len(eval_results) == 0:
            eval_results = None

        return eval_results

    def args_need_to_record(self) -> Dict[str, Any]:
        result = {'combine_two_text_as_input': self.model_args.combine_two_texts_as_input}
        result.update(super().args_need_to_record())
        return result



