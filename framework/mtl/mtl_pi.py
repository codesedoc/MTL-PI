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
from model import SemanticLayer, SemanticCompareEnum


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
    name = 'MTL_PI'
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

        self.max_token_length = int(self.framework_proxy.data_proxy.data_args.max_seq_length/2)
        self.semantic_layer = SemanticLayer(SemanticCompareEnum.l1)

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

        loss_weight = 0.4
        loss = loss_weight*a_loss + (1-loss_weight)*p_loss

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
        other_tfr_output = tfr_output[2:]
        tfr_output = tfr_output[0]
        texts_num_of_tokens_batch = input_['texts_num_of_tokens']
        if len(texts_num_of_tokens_batch) != len(tfr_output):
            raise ValueError

        attention_mask = input_['attention_mask']

        text_a_states_batch = []
        text_b_states_batch = []
        for a_m, t_output, texts_num_of_tokens in zip(attention_mask, tfr_output, texts_num_of_tokens_batch):
            sequen_len = a_m.sum()
            text_a_len = texts_num_of_tokens[0]
            text_b_len = texts_num_of_tokens[1]
            if text_a_len + text_b_len +3 != sequen_len:
                raise ValueError

            text_a_output_ = t_output[1: text_a_len+1]
            text_b_output_ = t_output[text_a_len+2: text_b_len+text_a_len+2]

            from utils.data_tool import padding_tensor
            text_a_output = padding_tensor(text_a_output_, self.max_token_length, align_dir='left', dim=0)
            text_b_output = padding_tensor(text_b_output_, self.max_token_length, align_dir='left', dim=0)

            text_a_states_batch.append(text_a_output)
            text_b_states_batch.append(text_b_output)

        text_a_states_batch = torch.stack(text_a_states_batch, dim=0)
        text_b_states_batch = torch.stack(text_b_states_batch, dim=0)

        tfrs_output = self.semantic_layer(text_a_states_batch, text_b_states_batch)
        tfrs_output = self.dropout(tfrs_output)

        if self.perform_state == PerformState.auxiliary:
            a_labels = input_.get('labels')
            if a_labels is None:
                raise ValueError

            result = self._single_forward(labels=a_labels, tfrs_output=tfrs_output)

        elif self.perform_state == PerformState.primary:
            p_labels = input_.get('labels')

            result = self._single_forward(labels=p_labels, tfrs_output=tfrs_output)

        elif self.perform_state == PerformState.parallel:
            p_labels = input_.get('labels')
            a_labels = input_.get('auxiliary_label')
            if a_labels is None or p_labels is None:
                raise ValueError

            result = self._parallel_forward(primary_labels=p_labels, auxiliary_labels=a_labels, tfrs_output=tfrs_output)
        else:
            raise ValueError

        outputs = result + other_tfr_output  # add hidden states and attention if they are here

        return outputs  # (loss), logits, (hidden_states), (attentions)


class MTLPIFrameworkProxy(TFRsFrameworkProxy):
    framework_class = MTLPIFramework

    def __init__(self, model_args: MTLPIModelArguments, performing_args: MTLPIPerformingArguments, data_proxy: MTLPIDataProxy,
                 *args, **kwargs):

        super().__init__(model_args, performing_args, data_proxy, *args, **kwargs)

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

        self.primary_performing_args = performing_args
        self.auxiliary_performing_args = replace(self.performing_args,
                                                 num_train_epochs=performing_args.auxiliary_training_epoch,
                                                 learning_rate=performing_args.auxiliary_learning_rate)

        self.performing_args = self.primary_performing_args

        self.primary_data_proxy = data_proxy.primary_sub_proxy

        self.auxiliary_data_proxy = data_proxy.auxiliary_sub_proxy

        self.primary_data_proxy.tokenizer = tokenizer
        self.auxiliary_data_proxy.tokenizer = tokenizer

        self.data_proxy = self.auxiliary_data_proxy

        self.framework:MTLPIFramework = self.framework
        # self.data_proxy.get_dataset(DataSetType.train)
        # self.data_proxy.get_dataset(DataSetType.dev)

    def _create_framework(self) -> MTLPIFramework:
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

    def _predict_for_primary_train(self):
        # predict_output = self._predict(DataSetType.train)
        from utils import file_tool
        id2pred = file_tool.load_data_pickle('utils/Hierarchic/mrpc_eid2elab_pred.pkl')

        return PredictionOutput(predictions=None, label_ids=None, metrics=None, indexes=None, example_id2pred=id2pred)

    def train(self,  *args, **kwargs):
        performing_args = self.performing_args

        # Training
        if performing_args.do_train:
            logger.info("*** Step1: Train auxiliary data ***")

            # self.framework.perform_state = PerformState.auxiliary
            #
            # self._switch_to_auxiliary_data()
            #
            # dataset = self.data_proxy.merge_datasets(ds_types=(DataSetType.train, DataSetType.test, DataSetType.dev))
            # self.data_proxy.set_datasets(DataSetType.train, dataset)
            # self._train()

            logger.info("*** Step2: Predict primary data ***")

            self._switch_to_primary_data()

            predict_output = self._predict_for_primary_train()

            updates = self._create_update_input_features_updates(predict_output)

            if not self.performing_args.skip_revise_predictions:
                updates, revise_details = self.original_data_proxy.revise_invalid_predict_for_primary_task(updates=updates)
                if self.tb_writer is not None:
                    import json
                    self.tb_writer.add_text("revise_details_about_predicted_label_by_auxiliary_model",
                                            json.dumps(revise_details, indent=2), global_step=self.global_step)

            self.data_proxy.update_inputfeatures_in_dataset(DataSetType.train, updates)

            # self.framework.primary_classifier.load_state_dict(self.framework.auxiliary_classifier.state_dict())

            logger.info("*** Step3: Parallel train ***")

            self.framework.perform_state = PerformState.parallel

            self._train()

            logger.info("*** Step4: Evaluate primary data ***")
            self._switch_to_primary_data()

            self.framework.perform_state = PerformState.primary
            # self.save_model()

    def _switch_to_auxiliary_data(self):
        self.performing_args = self.auxiliary_performing_args
        self.data_proxy = self.auxiliary_data_proxy

    def _switch_to_primary_data(self):
        self.performing_args = self.primary_performing_args
        self.data_proxy = self.primary_data_proxy

    def _evaluate_during_training(self):
        perform_state = self.framework.perform_state
        if perform_state == PerformState.auxiliary:
            # metrics = self._evaluate(DataSetType.test)
            # return metrics
            return None

        elif perform_state == PerformState.parallel or perform_state == PerformState.primary:
            self.framework.perform_state = PerformState.primary
            metrics = self._evaluate(DataSetType.dev)

            # test_metrics = self._evaluate(DataSetType.test)
            # print(test_metrics)
            self.framework.perform_state = PerformState.parallel
            return metrics
        else:
            raise ValueError

