from .. import Framework, FrameworkProxy
from argument.mtl_pi import MTLPIModelArguments, MTLPIPerformingArguments
from data.proxy.mtl_pi import MTLPIDataProxy
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from ..transformers._utils import AdamW, get_linear_schedule_with_warmup, is_apex_available
from typing import Any, Tuple, Dict, Optional, List, Union
from dataclasses import dataclass, replace
from data.proxy import DataSetType
from torch.utils.tensorboard import SummaryWriter
from  torch.utils.data import DataLoader, DistributedSampler, Dataset
import torch
import logging
logger = logging.getLogger(__name__)
from tqdm.auto import tqdm, trange
from ..transformers._utils import amp
from framework import TrainOutput, PredictionOutput
from packaging import version
import utils.file_tool as file_tool
import json
import numpy as np


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

        outputs = loss, ({'auxiliary': a_logits, 'primary': p_logits},)

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
            if p_labels is None:
                raise ValueError

            result = self._single_forward(labels=p_labels, tfrs_output=pooled_output)

        elif self.perform_state == PerformState.parallel:
            p_labels = input_.get('labels')
            a_labels = input_.get('auxiliary_labels')
            if a_labels is None or p_labels is None:
                raise ValueError

            result = self._parallel_forward(primary_labels=p_labels, auxiliary_labels=a_labels, tfrs_output=pooled_output)
        else:
            raise ValueError

        outputs = result + tfr_output[2:]  # add hidden states and attention if they are here

        return outputs  # (loss), logits, (hidden_states), (attentions)


class MTLPIFrameworkProxy(FrameworkProxy):
    EMPTY_PREFIX = ""

    def __init__(self, model_args: MTLPIModelArguments, performing_args: MTLPIPerformingArguments, data_proxy: MTLPIDataProxy,
                 *args, **kwargs):

        super().__init__(model_args, performing_args, data_proxy, *args, **kwargs)

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

        self.optimization_kit = None
        self.tb_writer = SummaryWriter(log_dir=performing_args.logging_dir)
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

        return MTLPIFramework(model_args, config=config)

        pass

    def _get_optimization_kit(self, num_total_training_steps: int) -> OptimizationKit:
        if self.optimization_kit is None:
            optimizer_grouped_parameters = self.framework.get_optimizer_grouped_parameters()
            args = self.performing_args
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_total_training_steps
            )
            self.optimization_kit = OptimizationKit(optimizer, scheduler)

        return self.optimization_kit

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
            logger.info("*** Step1: Train auxiliary data ***")

            dataset = self.data_proxy.merge_datasets(ds_types=(DataSetType.train, DataSetType.test, DataSetType.dev))
            self.data_proxy.set_datasets(DataSetType.train, dataset)
            self.train()

            logger.info("*** Step2: Predict primary data ***")
            self.performing_args = self.primary_performing_args
            self.data_proxy = self.primary_data_proxy

            predict_output = self._predict(DataSetType.train)
            updates = self._create_update_input_features_updates(predict_output)
            self.original_data_proxy.revise_invalid_predict_for_primary_task(updates=updates)
            self.data_proxy.update_inputfeatures_in_dataset(DataSetType.train, )

            self.framework.perform_state = PerformState.parallel
            self.train()
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

    def train(self):
        train_dataloader = self.data_proxy.get_dataloader(DataSetType.train)

        # from utils.general_tool import setup_seed
        # setup_seed(self.model_args.seed)
        args = self.performing_args
        if args.max_steps > 0:
            t_total = args.max_steps
            num_train_epochs = (
                    args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
            )
            logging_steps = args.logging_steps
        else:
            t_total = int(len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs)
            num_train_epochs = args.num_train_epochs
            logging_steps = len(train_dataloader)

        kit = self._get_optimization_kit(num_total_training_steps=t_total)
        optimizer, scheduler = kit.optimizer, kit.lr_scheduler

        model = self.framework

        if args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", args.to_json_string())
            self.tb_writer.add_hparams(args.to_sanitized_dict(), metric_dict={})

        # Train!

        total_train_batch_size = self.data_proxy.data_args.train_batch_size * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.data_proxy.get_num_examples(DataSetType.train))
        logger.info("  Num Epochs = %d", num_train_epochs)
        # logger.info("  Instantaneous batch size per device = %d", self.data_proxy.data_args.per_device_train_batch_size)
        # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Batch size = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        steps_trained_in_current_epoch = 0

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        # train_iterator = trange(0, int(num_train_epochs), desc="Epoch")
        train_iterator = range(int(num_train_epochs))
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)


            epoch_iterator = tqdm(train_dataloader, desc="Iteration")

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self._training_step(model, inputs, optimizer)

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                ):
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (logging_steps > 0 and self.global_step % logging_steps == 0) or (
                            self.global_step == 1 and args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss
                        self._log(logs)
                        if args.evaluate_during_training:
                            # logs.update(self._evaluate(DataSetType.dev))
                            # logs.update(self._evaluate(DataSetType.test))
                            self._evaluate(DataSetType.dev)
                            self._evaluate(DataSetType.test)


                    if args.save_steps > 0 and self.global_step % args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.framework
                        else:
                            assert model is self.framework
                        # Save model checkpoint
                        output_path = file_tool.connect_path(args.output_dir, f"checkpoint-{self.global_step}")
                        self.save_model(output_path)

                if args.max_steps > 0 and self.global_step > args.max_steps:
                    epoch_iterator.close()
                    break
            if args.max_steps > 0 and self.global_step > args.max_steps:
                # train_iterator.close()
                break

        if self.tb_writer:
            self.tb_writer.close()

        logger.info("\n\nTraining completed.\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step)

    def _log(self, logs: Dict[str, float]) -> None:
        if not hasattr(self, 'epoch'):
            self.epoch = self.performing_args.num_train_epochs
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        new_log = {f'{self._log_prefix}_{k}': v for k, v in logs.items()}
        if self.tb_writer:
            for k, v in new_log.items():
                self.tb_writer.add_scalar(k, v, self.global_step)
            self.tb_writer.flush()

        output = json.dumps({**new_log, **{"step": self.global_step}})

        logger.info(output)

    def _training_step(
            self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], optimizer: Optimizer
    ) -> float:
        model.train()
        args = self.performing_args
        for k, v in inputs.items():
            inputs[k] = v.to(args.device)

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def evaluate(self):
        args = self.performing_args
        data_args = self.data_proxy.data_args
        eval_results = {}

        def do_evaluate():

            eval_result = self._evaluate(DataSetType.dev)

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

    def _evaluate(
            self, ds_type: DataSetType,
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        if not self.data_proxy.has_label(ds_type):
            logger.error(f'The {ds_type.value} set has not labels, so it can not be evaluated')

        eval_dataloader = self.data_proxy.get_dataloader(ds_type)

        output = self._prediction_loop(eval_dataloader, description="Evaluation", ds_type=ds_type)

        self._log(output.metrics)

        return output.metrics

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

                output_eval_file = file_tool.connect_path(args.output_dir, f"eval_results_{data_args.task_name}.txt")

                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval test set results {} *****".format(data_args.task_name))
                    for key, value in metrics_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                eval_results.update(metrics_result)
                self._log(metrics_result)
            self._save_prediction(indexes=predict_output.indexes, predictions=predict_output.predictions)

        do_predict()
        if data_args.task_name == "mnli":
            replace(data_args, task_name="mnli-mm")
            do_predict()
            replace(data_args, task_name="mnli")

    def _predict(self, ds_type: Optional[DataSetType] = None) -> PredictionOutput:
        if ds_type is None:
            ds_type = DataSetType.test

        test_dataloader = self.data_proxy.get_dataloader(ds_type)

        return self._prediction_loop(test_dataloader, description="Prediction", ds_type=ds_type)

    # def save_model(self, output_path: Optional[str] = None):
    #     pass

    def _prediction_loop(
            self, dataloader: DataLoader, description: str, ds_type: DataSetType, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        # prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.framework
        args = self.performing_args
        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.framework
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.data_proxy.get_num_examples(ds_type))
        logger.info("  Batch size = %d", batch_size)
        losses: List[float] = []
        preds: Optional[Union[torch.Tensor, np.ndarray]] = None
        label_ids: Optional[Union[torch.Tensor, np.ndarray]] = None
        indexes:  Optional[Union[torch.Tensor, np.ndarray]] = None
        example_id2pred: Optional[Dict] = {}
        example_ids: Optional[Union[torch.Tensor, List[int]]] = None
        model.eval()

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                inputs[k] = v.to(args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_loss, logits = outputs[:2]
                    losses += [step_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)

                if inputs.get('example_id') is not None:
                    if example_ids is None:
                        example_ids = inputs["example_id"].detach()
                    else:
                        example_ids = torch.cat((example_ids, inputs["example_id"].detach()), dim=0)

                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach()
                    else:
                        label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)

                if inputs.get("index") is not None:
                    if indexes is None:
                        indexes = inputs["index"].detach()
                    else:
                        indexes = torch.cat((indexes, inputs["index"].detach()), dim=0)

        if preds is not None:
            preds = preds.cpu().numpy().copy()
            from data import OutputMode
            output_mode = self.data_proxy.corpus.output_mode
            if output_mode == OutputMode.classification:
                preds = np.argmax(preds, axis=1)
            elif output_mode == OutputMode.regression:
                preds = np.squeeze(preds)

            example_ids = example_ids.cpu().tolist().copy()
            if len(example_ids) != len(preds):
                raise ValueError
            for e_id, pred in zip(example_ids, preds):
                example_id2pred[e_id] = pred
            if len(example_ids) != len(example_id2pred):
                raise ValueError

        if label_ids is not None:
            label_ids = label_ids.cpu().numpy().copy()

        if indexes is not None:
            indexes = indexes.cpu().numpy().copy()

        if preds is not None and label_ids is not None:
            from data.corpus import ItemsForMetricsComputation
            metrics = self.data_proxy.compute_metrics(ItemsForMetricsComputation(predictions=preds, label_ids=label_ids))

        else:
            metrics = {}

        prefix = ds_type.value
        if len(losses) > 0:
            metrics[f"{prefix}_loss"] = np.mean(losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):

            if not key.startswith(f"{prefix}_"):
                metrics[f"{prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics, indexes=indexes, example_id2pred=example_id2pred)


