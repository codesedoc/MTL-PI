from .. import Framework, FrameworkProxy
from argument.mtl_pi import MTLPIModelArguments, MTLPIPerformingArguments
from data.proxy.mtl_pi import MTLPIDataProxy
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ..transformers.run_glue import TFRsFrameworkProxy, TransformerTypeEnum

from typing import Any, Tuple, Dict, Optional, List, Union
from dataclasses import dataclass, replace
from data.proxy import DataSetType
from model import SemanticLayer, DistanceTypeEnum
from argument.mtl_pi import ModeForTrainData
from .. import MiniBatch

class MTLMiniBatch(MiniBatch):
    def __init__(self, data=None):
        super().__init__(data)
        self.task = None

import torch
import logging
logger = logging.getLogger(__name__)

from framework import TrainOutput, PredictionOutput
from argument.mtl_pi import FeatureComparedEnum



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

        self.transformer_type = TransformerTypeEnum.get_enum_by_value(model_args.model_name_or_path)

        # input_size_of_classifier = config.hidden_size
        # if (model_args.split_two_texts_as_input) and model_args.distance_type == DistanceTypeEnum.dim_l1.value:
        #     input_size_of_classifier += 1

        transformer_type = self.transformer_type
        if transformer_type == TransformerTypeEnum.bert or transformer_type == TransformerTypeEnum.albert:
            self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
            self.auxiliary_classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
            self.primary_classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        elif transformer_type == TransformerTypeEnum.xlnet:
            from transformers.modeling_utils import SequenceSummary
            self.xlnet_squence_sum = SequenceSummary(config)
            self.xlnet_squence_sum.apply(self._init_weights)
            self.auxiliary_classifier = torch.nn.Linear(config.d_model, config.num_labels)
            self.primary_classifier = torch.nn.Linear(config.d_model, config.num_labels)
        else:
            self.auxiliary_classifier = transformer_type.value(config)
            self.primary_classifier = transformer_type.value(config)

        self.auxiliary_classifier.apply(self._init_weights)
        self.primary_classifier.apply(self._init_weights)

        if model_args.learn_calibrator_weight:
            self.calibrator_weight = torch.nn.Linear(config.hidden_size, 1)
            self.calibrator_weight.apply(self._init_weights)

        self.perform_state = PerformState.auxiliary
        self.num_labels = config.num_labels

        # self.max_token_length = int(self.framework_proxy.data_proxy.data_args.max_seq_length)
        # self.loss_weight = model_args.loss_a
        self.model_args = model_args

        # if not self.model_args.split_two_texts_as_input:
        #     self.model_args = replace(model_args, feature_compared='None', distance_type='None')
        #
        # self.semantic_layer = SemanticLayer(model_args.distance_type)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (torch.nn.Linear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            if self.transformer_type == TransformerTypeEnum.xlm:
                std = self.encoder.config.init_std
            else:
                std = self.encoder.config.initializer_range

            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def _calculate_loss(self, logits, labels, loss_fct):
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss

    def _parallel_forward(self, primary_labels, auxiliary_labels, features_classified):
        a_logits = self.auxiliary_classifier(features_classified)
        p_logits = self.primary_classifier(features_classified)

        if self.model_args.tune_off_auxiliary_when_parallel:
            a_logits = a_logits.detach()

        if self.model_args.learn_calibrator_weight:
            if self.transformer_type == TransformerTypeEnum.roberta or  self.transformer_type == TransformerTypeEnum.electra:
                weight = torch.sigmoid(self.calibrator_weight(features_classified[:, 0, :]))
            else:
                weight = torch.sigmoid(self.calibrator_weight(features_classified))
            logits = torch.bmm(a_logits.unsqueeze(dim=1),
                              (weight.unsqueeze(dim=1)*torch.tensor([[-1, 1], [0, 0]], device=a_logits.device).expand(a_logits.shape[0], -1, -1)
                               + torch.tensor([[1, 0], [1, 0]], device=a_logits.device).expand(a_logits.shape[0], -1, -1))
                              ).squeeze()\
                     + p_logits

        else:
            weight = self.model_args.calibrator_weight

            logits = torch.mm(a_logits, torch.tensor([[1-weight, weight], [1, 0]], device=a_logits.device)) + p_logits

        outputs = logits,
        if primary_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = self._calculate_loss(logits, primary_labels, loss_fct)

            outputs = loss, logits

        return outputs

    def _single_forward(self, labels, features_classified):
        if self.perform_state == PerformState.primary:
            logits = self.primary_classifier(features_classified)
        else:
            logits = self.auxiliary_classifier(features_classified)
        outputs = logits,
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = self._calculate_loss(logits=logits, labels=labels, loss_fct=loss_fct)
            outputs = loss, logits
        return outputs

    def forward(self, **input_):
        if not self.model_args.split_two_texts_as_input:
            tfr_input_name = ['input_ids', 'attention_mask', 'token_type_ids']
            tfr_input ={}
            for name in tfr_input_name:
                if name in input_:
                    tfr_input[name] = input_[name]

            tfr_output = self.encoder(**tfr_input)[0]

            transformer_type = self.transformer_type

            if transformer_type == TransformerTypeEnum.bert or transformer_type == TransformerTypeEnum.albert:
                features = tfr_output[:, 0]

                features = self.dropout(features)
            elif transformer_type == TransformerTypeEnum.xlnet:
                features = self.xlnet_squence_sum(tfr_output)
            else:
                features = tfr_output

        else:
            raise ValueError
            # tfr_input_name = ['input_ids', 'attention_mask']
            # tfr_input_for_first = {}
            # tfr_input_for_second = {}
            #
            # for name in tfr_input_name:
            #     if name in input_:
            #         tfr_input_for_first[name] = input_[name]
            #         tfr_input_for_second[name] = input_[f"second_{name}"]
            #
            # def states_batch_length_batch(tfr_input_):
            #     if self.model_args.feature_compared == FeatureComparedEnum.tokens.value:
            #         tfr_output_ = self.encoder(**tfr_input_)[0]
            #
            #         tfr_output_ = tfr_output_ * tfr_input_['attention_mask'][:,:,None]
            #         tfr_output_ = tfr_output_[:, 1:]
            #
            #         length_batch = tfr_input_['attention_mask'].sum(dim=1, keepdim=True) - 1
            #
            #     elif self.model_args.feature_compared == FeatureComparedEnum.cls.value:
            #         tfr_output_ = self.encoder(**tfr_input_)[0][:, 0]
            #         tfr_output_ = tfr_output_[:, None, :]
            #         length_batch = 1
            #     else:
            #         raise ValueError
            #
            #     tfr_output_ = self.dropout(tfr_output_)
            #     return tfr_output_, length_batch
            #
            # text_a_states_batch, text_a_length_batch = states_batch_length_batch(tfr_input_for_first)
            # text_b_states_batch, text_b_length_batch = states_batch_length_batch(tfr_input_for_second)
            #
            # features = self.semantic_layer(text_a_states_batch, text_b_states_batch, text_a_length_batch, text_b_length_batch)

        features_classified = features

        if self.perform_state == PerformState.auxiliary:
            a_labels = input_.get('labels')
            if a_labels is None:
                raise ValueError

            result = self._single_forward(labels=a_labels, features_classified=features_classified)

        elif self.perform_state == PerformState.primary:
            if self.framework_proxy.adjust_prediction and not self.framework_proxy.chose_two_way_when_evaluate:
                raise ValueError

            p_labels = input_.get('labels')

            result = self._single_forward(labels=p_labels, features_classified=features_classified)

        elif self.perform_state == PerformState.parallel:
            if not self.framework_proxy.adjust_prediction:
                raise ValueError
            p_labels = input_.get('labels')
            a_labels = input_.get('auxiliary_label')
            # if a_labels is None or p_labels is None:
            #     raise ValueError

            result = self._parallel_forward(primary_labels=p_labels, auxiliary_labels=a_labels, features_classified=features_classified)
        else:
            raise ValueError

        return result  # (loss), logits


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

        self.primary_data_proxy.combine_two_text_as_input = not model_args.split_two_texts_as_input
        self.auxiliary_data_proxy.combine_two_text_as_input = not model_args.split_two_texts_as_input

        self.data_proxy = self.auxiliary_data_proxy

        self.framework:MTLPIFramework = self.framework
        self.model_args = self.framework.model_args

        # self.chose_two_way_when_evaluate = False

        self.chose_two_way_when_evaluate = True

        self.adjust_prediction = model_args.adjust_prediction

        if not self.adjust_prediction:
            self.chose_two_way_when_evaluate = False

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

        framework = MTLPIFramework(model_args, config=config, framework_proxy=self)
        if not self.model_args.single_task and self.performing_args.train_mode == ModeForTrainData.delay:
            from utils import file_tool
            self.framework_path = file_tool.connect_path(file_tool.dirname(__file__), 'after_auxiliary',
                                                         f'auep_{self.performing_args.auxiliary_training_epoch}',
                                                         f'aulr_{self.performing_args.auxiliary_learning_rate}')
            self.pretrained_auxiliary = False

            if file_tool.check_dir(self.framework_path):
                framework = self.load_model(self.framework_path, framework=framework)
                logger.info(f'load framework from {self.framework_path}')
                self.pretrained_auxiliary = True
        return framework

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

    def __delay_train(self,  *args, **kwargs):
        logger.info("*** Step1: Train auxiliary data ***")

        if self.framework.model_args.single_task:
            logging.info(f'Skip train auxiliary data')
        else:
            if self.pretrained_auxiliary:
                logging.info(f'Already trained auxiliary data')
            else:

                self.framework.perform_state = PerformState.auxiliary
                self._switch_to_auxiliary_data()

                dataset = self.data_proxy.merge_datasets(
                    ds_types=(DataSetType.train, DataSetType.dev, DataSetType.test))
                self.data_proxy.set_datasets(DataSetType.train, dataset)
                self._train()
                self.save_model(self.framework_path)

        from utils.general_tool import setup_seed

        setup_seed(self.model_args.seed)

        logger.info("*** Step2: Train primary data ***")
        self._switch_to_primary_data()

        if self.adjust_prediction:
            self.framework.perform_state = PerformState.parallel
            logging.info(f'****************** Adjust prediction for primary task *******************')
        else:
            self.framework.perform_state = PerformState.primary
            logging.info(f'****************** Do not adjust prediction for primary task *******************')

        self._train()

        logger.info("*** Step3: Evaluate primary data ***")
        # self._switch_to_primary_data()

        # self.framework.perform_state = PerformState.primary

        # if self.chose_two_way_when_evaluate:
        #     self.framework.perform_state = PerformState.parallel
        #     logging.info(f'******************Chose two way*******************')
        # else:
        #     self.framework.perform_state = PerformState.primary
        #     logging.info(f'******************Chose single way*******************')

        # self.save_model()

    def __cross_train(self,  *args, **kwargs):
        logger.info("*** Cross Training data ***")
        self.data_proxy = self.primary_data_proxy
        dataset = self.auxiliary_data_proxy.merge_datasets(
            ds_types=(DataSetType.train, DataSetType.dev, DataSetType.test))
        self.auxiliary_data_proxy.set_datasets(DataSetType.train, dataset)
        self._train()

    def __mix_train(self,  *args, **kwargs):
        logger.info("*** Mix Training data ***")
        self.data_proxy = self.primary_data_proxy
        dataset = self.auxiliary_data_proxy.merge_datasets(
            ds_types=(DataSetType.train, DataSetType.dev, DataSetType.test))
        self.auxiliary_data_proxy.set_datasets(DataSetType.train, dataset)
        self._train()

    def train(self,  *args, **kwargs):
        if self.performing_args.train_mode == ModeForTrainData.delay:
            self.__delay_train(*args, **kwargs)
        elif self.performing_args.train_mode == ModeForTrainData.cross:
            self.__cross_train(*args, **kwargs)
        elif self.performing_args.train_mode == ModeForTrainData.mix:
            self.__mix_train(*args, **kwargs)
        else:
            raise ValueError


    def predict(self):
        self._switch_to_primary_data()

        if self.chose_two_way_when_evaluate:
            self.framework.perform_state = PerformState.parallel
            logging.info(f'******************Chose two way*******************')
        else:
            self.framework.perform_state = PerformState.primary
            logging.info(f'******************Chose single way*******************')

        result = super().predict()

        return result

    def evaluate(self):
        self._switch_to_primary_data()
        if self.chose_two_way_when_evaluate:
            self.framework.perform_state = PerformState.parallel
            logging.info(f'******************Chose two way*******************')
        else:
            self.framework.perform_state = PerformState.primary
            logging.info(f'******************Chose single way*******************')

        result = super().evaluate()
        return result

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

        elif perform_state == PerformState.parallel:
            if self.chose_two_way_when_evaluate:
                self.framework.perform_state = PerformState.parallel
                logging.info(f'******************Chose two way*******************')
            else:
                self.framework.perform_state = PerformState.primary
                logging.info(f'******************Chose single way*******************')

            metrics = self._evaluate(DataSetType.dev)

            # test_metrics = self._evaluate(DataSetType.test)
            # print(test_metrics)
            self.framework.perform_state = PerformState.parallel
            return metrics

        elif perform_state == PerformState.primary and not self.adjust_prediction:
            logging.info(f'******************Chose single way*******************')
            metrics = self._evaluate(DataSetType.dev)
            return metrics

        else:
            raise ValueError

    def args_need_to_record(self) -> Dict[str, Any]:
        result = {
            'transformer': f"{self.framework.transformer_type.name}: {self.model_args.model_name_or_path}",
            # 'distance_type': self.model_args.distance_type,
            # 'feature_compared': self.model_args.feature_compared,
            'train_mode': self.performing_args.train_mode.value,
            'chose_two_way_when_evaluate': self.chose_two_way_when_evaluate,
            'adjust_prediction': self.model_args.adjust_prediction,
            'single_task': self.model_args.single_task,
            'calibrator_weight': self.model_args.calibrator_weight
        }

        result.update(super().args_need_to_record())
        return result

    def _get_mini_batches(self):
        result = []
        if self.performing_args.train_mode == ModeForTrainData.delay:
            result = super()._get_mini_batches()
        else:
            for x in list(self.auxiliary_data_proxy.get_dataloader(DataSetType.train)):
                temp = MTLMiniBatch(x)
                temp.task = "auxiliary"
                result.append(temp)

            for x in list(self.primary_data_proxy.get_dataloader(DataSetType.train)):
                temp = MTLMiniBatch(x)
                temp.task = "primary"
                result.append(temp)

            if self.performing_args.train_mode == ModeForTrainData.mix:
                import random
                random.shuffle(result)

        return result

    def _train_step(
            self, model: torch.nn.Module, mini_batch: MTLMiniBatch, optimizer: Optimizer
    ) -> float:
        if self.performing_args.train_mode != ModeForTrainData.delay:
            if mini_batch.task == "auxiliary":
                self.framework.perform_state = PerformState.auxiliary

            elif mini_batch.task == "primary":
                self.framework.perform_state = PerformState.parallel
            else:
                raise NotImplementedError()

        return super()._train_step(model, mini_batch, optimizer)
        # return 0

    def _get_num_train_exampels(self):
        primary_num = self.primary_data_proxy.get_num_examples(DataSetType.train)
        auxiliary_num = self.auxiliary_data_proxy.get_num_examples(DataSetType.train)
        if self.performing_args.train_mode == ModeForTrainData.delay:
            if self.framework.perform_state == PerformState.auxiliary:
                result = auxiliary_num
            elif self.framework.perform_state == PerformState.primary:
                result = primary_num
            elif self.framework.perform_state == PerformState.parallel:
                result = primary_num
            else:
                raise ValueError

        elif self.performing_args.train_mode == ModeForTrainData.cross:
            result = primary_num + auxiliary_num
        elif self.performing_args.train_mode == ModeForTrainData.mix:
            result = primary_num + auxiliary_num
        else:
            raise ValueError
        return result