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
    def __init__(self, model_args: TFRsModelArguments, performing_args: TFRsPerformingArguments, data_proxy: TFRsDataProxy,
                 *args, **kwargs):
        self.data_proxy = data_proxy

        self.data_proxy.tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

        super().__init__(model_args, performing_args, data_proxy, *args, **kwargs)

        self.optimization_kit = None
        self.tb_writer = SummaryWriter(log_dir=performing_args.logging_dir)
        self.model_args = model_args
        self.performing_args = performing_args
        self.global_step: Optional[int] = None
        self.epoch: Optional[float] = None
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

    # def train(self):
    #     train_dataloader = self.data_proxy.get_dataloader(DataSetType.train)
    #
    #     # from utils.general_tool import setup_seed
    #     # setup_seed(self.model_args.seed)
    #     args = self.performing_args
    #     if args.max_steps > 0:
    #         t_total = args.max_steps
    #         num_train_epochs = (
    #                 args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    #         )
    #         logging_steps = args.logging_steps
    #     else:
    #         t_total = int(len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs)
    #         num_train_epochs = args.num_train_epochs
    #         logging_steps = len(train_dataloader)
    #
    #     kit = self._get_optimization_kit(num_total_training_steps=t_total)
    #     optimizer, scheduler = kit.optimizer, kit.lr_scheduler
    #
    #     model = self.framework
    #
    #     if args.fp16:
    #         if not is_apex_available():
    #             raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #         model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    #
    #     # multi-gpu training (should be after apex fp16 initialization)
    #     if args.n_gpu > 1:
    #         model = torch.nn.DataParallel(model)
    #
    #     if self.tb_writer is not None:
    #         self.tb_writer.add_text("args", args.to_json_string())
    #         self.tb_writer.add_hparams(args.to_sanitized_dict(), metric_dict={})
    #
    #     # Train!
    #
    #     total_train_batch_size = self.data_proxy.data_args.train_batch_size * args.gradient_accumulation_steps
    #
    #     logger.info("***** Running training *****")
    #     logger.info("  Num examples = %d", self.data_proxy.get_num_examples(DataSetType.train))
    #     logger.info("  Num Epochs = %d", num_train_epochs)
    #     # logger.info("  Instantaneous batch size per device = %d", self.data_proxy.data_args.per_device_train_batch_size)
    #     # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
    #     logger.info("  Batch size = %d", total_train_batch_size)
    #     logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    #     logger.info("  Total optimization steps = %d", t_total)
    #
    #     self.global_step = 0
    #     self.epoch = 0
    #     steps_trained_in_current_epoch = 0
    #
    #     tr_loss = 0.0
    #     logging_loss = 0.0
    #     model.zero_grad()
    #     # train_iterator = trange(0, int(num_train_epochs), desc="Epoch")
    #     train_iterator = range(int(num_train_epochs))
    #     for epoch in train_iterator:
    #         if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
    #             train_dataloader.sampler.set_epoch(epoch)
    #
    #
    #         epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    #
    #         for step, inputs in enumerate(epoch_iterator):
    #
    #             # Skip past any already trained steps if resuming training
    #             if steps_trained_in_current_epoch > 0:
    #                 steps_trained_in_current_epoch -= 1
    #                 continue
    #
    #             tr_loss += self._training_step(model, inputs, optimizer)
    #
    #             if (step + 1) % args.gradient_accumulation_steps == 0 or (
    #                     # last step in epoch but step is always smaller than gradient_accumulation_steps
    #                     len(epoch_iterator) <= args.gradient_accumulation_steps
    #                     and (step + 1) == len(epoch_iterator)
    #             ):
    #                 if args.fp16:
    #                     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
    #                 else:
    #                     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #
    #                     optimizer.step()
    #
    #                 scheduler.step()
    #                 model.zero_grad()
    #                 self.global_step += 1
    #                 self.epoch = epoch + (step + 1) / len(epoch_iterator)
    #
    #                 if (logging_steps > 0 and self.global_step % logging_steps == 0) or (
    #                         self.global_step == 1 and args.logging_first_step
    #                 ):
    #                     logs: Dict[str, float] = {}
    #                     logs["loss"] = (tr_loss - logging_loss) / logging_steps
    #                     # backward compatibility for pytorch schedulers
    #                     logs["learning_rate"] = (
    #                         scheduler.get_last_lr()[0]
    #                         if version.parse(torch.__version__) >= version.parse("1.4")
    #                         else scheduler.get_lr()[0]
    #                     )
    #                     logging_loss = tr_loss
    #                     self._log(logs)
    #                     if args.evaluate_during_training:
    #                         # logs.update(self._evaluate(DataSetType.dev))
    #                         # logs.update(self._evaluate(DataSetType.test))
    #                         self._evaluate(DataSetType.dev)
    #                         self._evaluate(DataSetType.test)
    #
    #
    #                 if args.save_steps > 0 and self.global_step % args.save_steps == 0:
    #                     # In all cases (even distributed/parallel), self.model is always a reference
    #                     # to the model we want to save.
    #                     if hasattr(model, "module"):
    #                         assert model.module is self.framework
    #                     else:
    #                         assert model is self.framework
    #                     # Save model checkpoint
    #                     output_path = file_tool.connect_path(args.output_dir, f"checkpoint-{self.global_step}")
    #                     self.save_model(output_path)
    #
    #             if args.max_steps > 0 and self.global_step > args.max_steps:
    #                 epoch_iterator.close()
    #                 break
    #         if args.max_steps > 0 and self.global_step > args.max_steps:
    #             # train_iterator.close()
    #             break
    #
    #     if self.tb_writer:
    #         self.tb_writer.close()
    #
    #     logger.info("\n\nTraining completed.\n\n")
    #     return TrainOutput(self.global_step, tr_loss / self.global_step)

    # def _log(self, logs: Dict[str, float]) -> None:
    #     if not hasattr(self, 'epoch'):
    #         self.epoch = self.performing_args.num_train_epochs
    #     if self.epoch is not None:
    #         logs["epoch"] = self.epoch
    #     if self.tb_writer:
    #         for k, v in logs.items():
    #             self.tb_writer.add_scalar(k, v, self.global_step)
    #         self.tb_writer.flush()
    #
    #     output = json.dumps({**logs, **{"step": self.global_step}})
    #
    #     logger.info(output)

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

    # def _evaluate(
    #         self, ds_type: DataSetType,
    # ) -> Dict[str, float]:
    #     """
    #     Run evaluation and return metrics.
    #
    #     The calling script will be responsible for providing a method to compute metrics, as they are
    #     task-dependent.
    #
    #     Args:
    #         eval_dataset: (Optional) Pass a dataset if you wish to override
    #         the one on the instance.
    #     Returns:
    #         A dict containing:
    #             - the eval loss
    #             - the potential metrics computed from the predictions
    #     """
    #     if not self.data_proxy.has_label(ds_type):
    #         logger.error(f'The {ds_type.value} set has not labels, so it can not be evaluated')
    #
    #     eval_dataloader = self.data_proxy.get_dataloader(ds_type)
    #
    #     output = self._prediction_loop(eval_dataloader, description="Evaluation", ds_type=ds_type)
    #
    #     self._log(output.metrics)
    #
    #     return output.metrics

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
                self._tensorboard_log(metrics_result)
            self._save_prediction(indexes=predict_output.indexes, predictions=predict_output.predictions)

        do_predict()
        if data_args.task_name == "mnli":
            replace(data_args, task_name="mnli-mm")
            do_predict()
            replace(data_args, task_name="mnli")

    # def _predict(self) -> PredictionOutput:
    #     test_dataloader = self.data_proxy.get_dataloader(DataSetType.test)
    #
    #     return self._prediction_loop(test_dataloader, description="Prediction", ds_type=DataSetType.test)

    # def save_model(self, output_path: Optional[str] = None):
    #     pass

    # def _prediction_loop(
    #         self, dataloader: DataLoader, description: str, ds_type: DataSetType, prediction_loss_only: Optional[bool] = None
    # ) -> PredictionOutput:
    #     """
    #     Prediction/evaluation loop, shared by `evaluate()` and `predict()`.
    #
    #     Works both with or without labels.
    #     """
    #
    #     # prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only
    #
    #     model = self.framework
    #     args = self.performing_args
    #     # multi-gpu eval
    #     if args.n_gpu > 1:
    #         model = torch.nn.DataParallel(model)
    #     else:
    #         model = self.framework
    #     # Note: in torch.distributed mode, there's no point in wrapping the model
    #     # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
    #
    #     batch_size = dataloader.batch_size
    #     logger.info("***** Running %s *****", description)
    #     logger.info("  Num examples = %d", self.data_proxy.get_num_examples(ds_type))
    #     logger.info("  Batch size = %d", batch_size)
    #     losses: List[float] = []
    #     preds: Optional[Union[torch.Tensor, np.ndarray]] = None
    #     label_ids: Optional[Union[torch.Tensor, np.ndarray]] = None
    #     indexes:  Optional[Union[torch.Tensor, np.ndarray]] = None
    #     model.eval()
    #
    #     for inputs in tqdm(dataloader, desc=description):
    #         has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])
    #
    #         for k, v in inputs.items():
    #             inputs[k] = v.to(args.device)
    #
    #         with torch.no_grad():
    #             outputs = model(**inputs)
    #             if has_labels:
    #                 step_loss, logits = outputs[:2]
    #                 losses += [step_loss.mean().item()]
    #             else:
    #                 logits = outputs[0]
    #
    #         if not prediction_loss_only:
    #             if preds is None:
    #                 preds = logits.detach()
    #             else:
    #                 preds = torch.cat((preds, logits.detach()), dim=0)
    #
    #             if inputs.get("labels") is not None:
    #                 if label_ids is None:
    #                     label_ids = inputs["labels"].detach()
    #                 else:
    #                     label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)
    #
    #             if inputs.get("index") is not None:
    #                 if indexes is None:
    #                     indexes = inputs["index"].detach()
    #                 else:
    #                     indexes = torch.cat((indexes, inputs["index"].detach()), dim=0)
    #
    #     if preds is not None:
    #         preds = preds.cpu().numpy().copy()
    #         from data import OutputMode
    #         output_mode = self.data_proxy.corpus.output_mode
    #         if output_mode == OutputMode.classification:
    #             preds = np.argmax(preds, axis=1)
    #         elif output_mode == OutputMode.regression:
    #             preds = np.squeeze(preds)
    #
    #     if label_ids is not None:
    #         label_ids = label_ids.cpu().numpy().copy()
    #
    #     if indexes is not None:
    #         indexes = indexes.cpu().numpy().copy()
    #
    #     if preds is not None and label_ids is not None:
    #         from data.corpus import ItemsForMetricsComputation
    #         metrics = self.data_proxy.compute_metrics(ItemsForMetricsComputation(predictions=preds, label_ids=label_ids))
    #
    #     else:
    #         metrics = {}
    #
    #     prefix = ds_type.value
    #     if len(losses) > 0:
    #         metrics[f"{prefix}_loss"] = np.mean(losses)
    #
    #     # Prefix all keys with eval_
    #     for key in list(metrics.keys()):
    #
    #         if not key.startswith(f"{prefix}_"):
    #             metrics[f"{prefix}_{key}"] = metrics.pop(key)
    #
    #     return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics, indexes=indexes)


