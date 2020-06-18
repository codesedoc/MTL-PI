from .. import Framework, FrameworkProxy
from argument.mtl_pi import MTLPIModelArguments, MTLPIPerformingArguments
from data.proxy.mtl_pi import MTLPIDataProxy
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ..transformers.run_glue import TFRsFrameworkProxy

from typing import Any, Tuple, Dict, Optional, List, Union
from dataclasses import dataclass, replace
from data.proxy import DataSetType

import torch
import logging
logger = logging.getLogger(__name__)

from framework import TrainOutput, PredictionOutput



@dataclass
class OptimizationKit:
    optimizer: Optimizer
    lr_scheduler: LambdaLR


from enum import Enum,unique


# class PredictionOutput(NamedTuple):
#     predictions: np.ndarray
#     label_ids: Optional[np.ndarray]
#     metrics: Optional[Dict[str, float]]
#     indexes: Optional[np.ndarray]


@unique
class PerformState(Enum):
    auxiliary = 'auxiliary'
    parallel = 'parallel'
    primary = 'primary'


class MTLPIFramework(Framework):

    def __init__(self, model_args: MTLPIModelArguments, *args, **kwargs):
        super().__init__(model_args, *args, **kwargs)
        config = kwargs['config']
        self.framework_proxy = kwargs['framework_proxy']
        self.encoder = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.auxiliary_classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.primary_classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.auxiliary_classifier.apply(self._init_weights)
        self.primary_classifier.apply(self._init_weights)
        self.perform_state = PerformState.auxiliary
        self.num_labels = config.num_labels

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (torch.nn.Linear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.encoder.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def _calculate_loss(self, logits, labels, loss_fct):
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss

    def _parallel_forward(self, primary_labels, auxiliary_labels, tfrs_output):
        a_logits = self.auxiliary_classifier(tfrs_output)
        p_logits = self.primary_classifier(tfrs_output)

        loss_fct = torch.nn.CrossEntropyLoss()
        a_loss = self._calculate_loss(a_logits, auxiliary_labels, loss_fct)
        p_loss = self._calculate_loss(p_logits, primary_labels, loss_fct)

        loss = (a_loss + p_loss).mean(dim=-1)

        outputs = loss, ({'auxiliary': a_logits, 'primary': p_logits}, (a_loss, p_loss))

        return outputs

    def _single_forward(self, labels, tfrs_output):
        if self.perform_state == PerformState.primary:
            logits = self.primary_classifier(tfrs_output)
        else:
            logits = self.auxiliary_classifier(tfrs_output)
        outputs = logits,
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = self._calculate_loss(logits=logits, labels=labels, loss_fct=loss_fct)
            outputs = loss, logits
        return outputs

    def forward(self, **input_):
        tfrs_input = {}
        for name in ['input_ids', 'attention_mask', 'token_type_ids']:
            if name in input_:
                tfrs_input[name] = input_[name]
        tfr_output = self.encoder(**tfrs_input)
        pooled_output = tfr_output[1]

        pooled_output = self.dropout(pooled_output)

        if self.perform_state == PerformState.auxiliary:
            a_labels = input_.get('labels')
            if a_labels is None:
                raise ValueError

            result = self._single_forward(labels=a_labels, tfrs_output=pooled_output)

        elif self.perform_state == PerformState.primary:
            p_labels = input_.get('labels')

            result = self._single_forward(labels=p_labels, tfrs_output=pooled_output)

        elif self.perform_state == PerformState.parallel:
            p_labels = input_.get('labels')
            a_labels = input_.get('auxiliary_label')
            if a_labels is None or p_labels is None:
                raise ValueError

            result = self._parallel_forward(primary_labels=p_labels, auxiliary_labels=a_labels, tfrs_output=pooled_output)
        else:
            raise ValueError

        outputs = result + tfr_output[2:]  # add hidden states and attention if they are here

        return outputs  # (loss), logits, (hidden_states), (attentions)


class MTLPIFrameworkProxy(TFRsFrameworkProxy):
    def __init__(self, model_args: MTLPIModelArguments, performing_args: MTLPIPerformingArguments, data_proxy: MTLPIDataProxy,
                 *args, **kwargs):

        super().__init__(model_args, performing_args, data_proxy, *args, **kwargs)

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

        self.optimization_kit = None
        self.model_args = model_args
        self.primary_performing_args = performing_args
        self.auxiliary_performing_args = replace(self.performing_args,
                                                 num_train_epochs=performing_args.auxiliary_training_epoch,
                                                 learning_rate=performing_args.learning_rate)

        self.performing_args = self.primary_performing_args

        self.global_step: Optional[int] = None
        self.epoch: Optional[float] = None

        self.primary_data_proxy = data_proxy.primary_sub_proxy
        self.auxiliary_data_proxy = data_proxy.auxiliary_sub_proxy

        self.primary_data_proxy.tokenizer = tokenizer
        self.auxiliary_data_proxy.tokenizer = tokenizer
        self._log_prefix = self.EMPTY_PREFIX
        self.original_data_proxy = data_proxy
        # self.data_proxy.get_dataset(DataSetType.train)
        # self.data_proxy.get_dataset(DataSetType.dev)

    def _create_framework(self)->MTLPIFramework:
        model_args = self.model_args

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=self.data_proxy.num_label,
            finetuning_task=self.data_proxy.task_name,
            cache_dir=model_args.cache_dir,
        )

        return MTLPIFramework(model_args, config=config, framework_proxy=self)

        pass

    def _create_update_input_features_updates(self, predict_output: PredictionOutput):
        from data import InputFeaturesUpdate
        e_id2pred = predict_output.example_id2pred
        updates = []
        for e_id, pred in e_id2pred.items():
            updates.append(InputFeaturesUpdate(self.primary_data_proxy, DataSetType.train, e_id, self.original_data_proxy.get_feature_field_for_predict(pred)))
        return updates

    def perform(self,  *args, **kwargs):
        performing_args = self.performing_args

        # Training
        if performing_args.do_train:
            self.framework.perform_state = PerformState.auxiliary
            self.performing_args = self.auxiliary_performing_args
            self.data_proxy = self.auxiliary_data_proxy
            self.performing_args.evaluate_during_training = False
            logger.info("*** Step1: Train auxiliary data ***")

            dataset = self.data_proxy.merge_datasets(ds_types=(DataSetType.train, DataSetType.test, DataSetType.dev))
            self.data_proxy.set_datasets(DataSetType.train, dataset)
            self.train()

            logger.info("*** Step2: Predict primary data ***")
            self.performing_args = self.primary_performing_args
            self.data_proxy = self.primary_data_proxy

            predict_output = self._predict(DataSetType.train)
            updates = self._create_update_input_features_updates(predict_output)
            # updates = self.original_data_proxy.revise_invalid_predict_for_primary_task(updates=updates)
            self.data_proxy.update_inputfeatures_in_dataset(DataSetType.train, updates)

            self.framework.perform_state = PerformState.parallel
            temp_flag = self.performing_args.evaluate_during_training

            self.performing_args.evaluate_during_training = False
            self.train()
            self.performing_args.evaluate_during_training = temp_flag

            # self.save_model()


        # self.data_proxy = self.primary_data_proxy
        # Evaluation
        self.framework.perform_state = PerformState.primary
        eval_results = {}
        if performing_args.do_eval:
            logger.info("*** Evaluate ***")
            eval_results = self.evaluate()

        # Prediction
        if performing_args.do_predict:
            logging.info("*** Test ***")
            self.predict()

        return eval_results


