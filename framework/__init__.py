from argument import ModelArguments, PerformingArguments
from typing import Optional, Tuple, Any, Dict, NamedTuple, List
import torch
import numpy as np
from torch.optim.optimizer import Optimizer
import utils.file_tool as file_tool
from utils.general_tool import setup_seed
from data.proxy import DataProxy
from torch.utils.data import DataLoader, DistributedSampler
from data import DataSetType
from typing import Union
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from torch.optim.lr_scheduler import LambdaLR
from ._utils import is_apex_available, amp

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


@dataclass
class OptimizationKit:
    optimizer: Optimizer
    lr_scheduler: LambdaLR


class Framework(torch.nn.Module):
    parameter_name_for_no_decay = ["bias"]
    name = 'framework'
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
    EMPTY_PREFIX = ""
    framework_class = Framework

    def __init__(self, model_args: ModelArguments, performing_args: PerformingArguments, data_proxy: DataProxy,
                 *args, **kwargs):
        self.model_args = model_args
        self.performing_args = performing_args
        self.data_proxy = data_proxy
        self.original_data_proxy = data_proxy
        # setup_seed(model_args.seed)

        self.framework = self._create_framework()
        self.framework.to(self.performing_args.device)

        self.tb_writer = None

        self.optimization_kit = None
        self.model_args = model_args
        self.performing_args = performing_args
        self.global_step: Optional[int] = None
        self.epoch: Optional[float] = None

        setup_seed(self.model_args.seed)

        pass


    @property
    def tensorboard_path(self):
        from socket import gethostname
        return file_tool.connect_path('result/tensorboard', self.framework.name, gethostname())

    @property
    def optuna_path(self):
        from socket import gethostname
        return file_tool.connect_path('result/optuna', self.framework.name, gethostname())

    def _create_framework(self) -> Framework:
        raise NotImplementedError()

    def _get_optimizers(self, num_total_training_steps: int) -> Any:
        raise NotImplementedError()

    def perform(self,  *args, **kwargs):
        self.tb_writer = SummaryWriter(log_dir=self.performing_args.logging_dir)
        self.tb_writer.add_text("performing_args", self.performing_args.to_json_string())
        self.tb_writer.add_text("model_args", self.model_args.to_json_string())
        self.tb_writer.add_text("data_args", self.original_data_proxy.data_args.to_json_string())

        performing_args = self.performing_args

        # Training
        if performing_args.do_train:
            logger.info("*** Train ***")
            self.train()
            # self.save_model()

        # print(torch.randn((5)))
        # Evaluation
        eval_results = {}
        if performing_args.do_eval:
            logger.info("*** Evaluate ***")
            eval_results = self.evaluate()

        # Prediction
        if performing_args.do_predict:
            logging.info("*** Test ***")
            self.predict()

        result = {
            'eval_results': eval_results
        }

        return result

    def _get_optimization_kit(self,  num_total_training_steps, force: bool = False,  **kwargs):
        raise NotImplementedError()

    def train(self, *args, **kwargs):
        self._train(*args, **kwargs)

    def _train(self, *args, **kwargs):
        from data import DataSetType
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

        kit = self._get_optimization_kit(num_total_training_steps=t_total, force=True)
        optimizer, scheduler = kit.optimizer, kit.lr_scheduler

        model = self.framework

        if args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

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

                tr_loss += self._train_step(model, inputs, optimizer)

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
                        from packaging import version
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss

                        added_logs = self._add_logs_when_train()
                        if added_logs is not None:
                            logs.update(added_logs)
                        self._tensorboard_log(logs=logs)

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

    def _evaluate_during_training(self):
        return self._evaluate(DataSetType.dev)

    def _add_logs_when_train(self):
        # if self.epoch == self.performing_args.num_train_epochs:
        #     return None

        if self.performing_args.evaluate_during_training:
            metrics = self._evaluate_during_training()
            return metrics
        return None

    def _train_step(
            self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], optimizer: Optimizer
    ) -> float:
        model.train()
        args = self.performing_args
        for k, v in inputs.items():
            inputs[k] = v.to(args.device)

        loss = self._calculate_loss(model, inputs)

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

    def _calculate_loss(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
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

    def load_model(self, model_path):
        model_file = file_tool.connect_path(model_path, 'framework.pt')
        if not file_tool.is_file(model_file):
            raise ValueError
        self.framework.load_state_dict(torch.load(model_file))
        logger.info("Load model from: %s", model_file)
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
        # for inputs in tqdm(dataloader, desc=description):
        for inputs in iter(dataloader):

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

    def _tensorboard_log(self, logs: Dict[str, float], prefix:str = None, step: int = None) -> None:
        tb_writer = self.tb_writer
        if step is None:
            step = self.global_step
        if prefix is None:
            prefix = self.data_proxy.task_name

        if not hasattr(self, 'epoch'):
            self.epoch = self.performing_args.num_train_epochs

        if self.epoch is not None:
            logs["epoch"] = self.epoch

        new_log = {f'{prefix}_{k}': v for k, v in logs.items()}
        if tb_writer:
            for k, v in new_log.copy().items():
                from utils.general_tool import is_number
                if not is_number(str(v)):
                    logger.warning(f"'{k}' is not scalar, so it can not be recorded by tensorboard")
                    new_log.pop(k)
                else:
                    tb_writer.add_scalar(k, v, step)
            tb_writer.flush()

        import json
        output = json.dumps({**new_log, **{"step": step}})

        print(output)

    def _evaluate(self, ds_type: DataSetType) -> Dict[str, float]:
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

        return output.metrics

    def _predict(self, ds_type: Optional[DataSetType] = None) -> PredictionOutput:
        if ds_type is None:
            ds_type = DataSetType.test

        test_dataloader = self.data_proxy.get_dataloader(ds_type)

        return self._prediction_loop(test_dataloader, description="Prediction", ds_type=ds_type)


def _create_framework_proxy(proxy_type: type, model_args: ModelArguments, performing_args: PerformingArguments,
                            *args, **kwargs) -> FrameworkProxy:

    framework_proxy = proxy_type(model_args, performing_args, *args, **kwargs)
    return framework_proxy


framework_proxy_singleton: Optional[FrameworkProxy] = None


def create_framework_proxy(proxy_type: type, model_args: ModelArguments, performing_args: PerformingArguments,
                           *args, **kwargs) -> FrameworkProxy:

    global framework_proxy_singleton
    force = kwargs.get('force') if kwargs.get('force') is not None else False
    if framework_proxy_singleton is None or force:
        framework_proxy_singleton = _create_framework_proxy(proxy_type, model_args, performing_args, *args, **kwargs)
    return framework_proxy_singleton
