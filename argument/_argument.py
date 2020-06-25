from dataclasses import dataclass, field, fields, asdict, replace
import dataclasses
from data.corpus import name2corpus_type
from typing import Optional,Tuple, Dict, Any
import utils.file_tool as file_tool
import logging
from functools import wraps
import torch,json
logger = logging.getLogger(__name__)


def default_tensorboard_dir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    # return file_tool.connect_path("runs", current_time + "_" + socket.gethostname())

    return file_tool.connect_path("runs", "_" + socket.gethostname())

class cached_property(property):
    """
    Descriptor that mimics @property but caches output in member variable.

    From tensorflow_datasets

    Built-in in functools from Python 3.8.
    """

    def __get__(self, obj, objtype=None):
        # See docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            cached = self.fget(obj)
            setattr(obj, attr, cached)
        return cached


try:
    import torch
    _torch_available = True  # pylint: disable=invalid-name
    logger.info("PyTorch version {} available.".format(torch.__version__))

except ImportError:
    _torch_available = False


def is_torch_available():
    return _torch_available


def torch_required(func):
    # Chose a different decorator name than in tests so it's clear they are not the same.
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_torch_available():
            return func(*args, **kwargs)
        else:
            raise ImportError(f"Method `{func.__name__}` requires PyTorch.")

    return wrapper


@dataclass(frozen=True)
class Arguments:
    field_name_replace_dict = {}
    pass

    @classmethod
    def field_names(self):
        if not isinstance(self, Arguments):
            _field_names = tuple([f.name for f in fields(self)])
        else:
            _field_names = tuple([f.name for f in asdict(self)])
        return _field_names

    @classmethod
    def search_init_kwargs(self, kwargs: Dict[str, Any]):
        _field_names = self.field_names()

        result = {}
        for name in _field_names:
            if name not in kwargs:
                raise ValueError
            else:
                result[name] = kwargs[name]
        return result

    @property
    def names2value(self):
        # names = self.field_names()
        # return {name: getattr(self, name) for name in names}
        return asdict(self)

    def get_name_abbreviation(self) -> Dict[str, Any]:
        return {}

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = dataclasses.asdict(self)
        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}

    def replace_args(self, replace_dict: Dict[str, Any]):
        def pick_up_arg(args_: Arguments):
            piched_args = {}
            arg_names2value = args_.names2value
            for name in replace_dict.copy():
                if name in arg_names2value:
                    piched_args[name] = replace_dict.pop(name)
            return piched_args

        result = self
        re_args = pick_up_arg(self)
        if len(re_args) > 0:
            new_args_obj = replace(self, **re_args)
            result = new_args_obj
        return result

@dataclass(frozen=True)
class ModelArguments(Arguments):
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    seed: int = field(default=1234, metadata={"help": "random seed for initialization"})

    def __post_init__(self):
        pass


@dataclass(frozen=True)
class DataArguments(Arguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(name2corpus_type.keys())})

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    overwrite_data_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    # config the arguments about batch_size
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/CPU for evaluation."}
    )

    @property
    def train_batch_size(self) -> int:
        return self.per_device_train_batch_size * max(1, torch.cuda.device_count())

    @property
    def eval_batch_size(self) -> int:
        return self.per_device_eval_batch_size * max(1, torch.cuda.device_count())

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )

    def __post_init__(self):
        pass

    def get_name_abbreviation(self):
        base_result = super().get_name_abbreviation()
        result = {
            "per_device_train_batch_size": "pdbz",
        }
        result.update(base_result)
        return result

@dataclass(frozen=True)
class PerformingArguments(Arguments):
    """
    TrainingArguments is the subset of the arguments we use in our example scripts
    **which relate to the training loop itself**.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    # config the arguments about the output
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions, evaluation results, system logs\
                          and checkpoints will be written."}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    # config the arguments for the procedure of running
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    evaluate_during_training: bool = field(
        default=False, metadata={"help": "Run evaluation during training at each logging step of model."},
    )

    # config the arguments for devices
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})

    # @cached_property
    @property
    @torch_required
    def _setup_devices(self) -> Tuple["torch.device", int]:
        if self.no_cuda:
            device = torch.device("cpu")
            n_gpu = 0

        else:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()

        return device, n_gpu

    @property
    @torch_required
    def device(self) -> "torch.device":
        return self._setup_devices[0]

    @property
    @torch_required
    def n_gpu(self):
        return self._setup_devices[1]

    # config the arguments about the updating of parameters of model
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    # config the arguments about the epoch and step
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    # config the arguments about saving during training
    logging_dir: Optional[str] = field(default_factory=default_tensorboard_dir, metadata={"help": "Tensorboard log dir."})
    logging_first_step: bool = field(default=False, metadata={"help": "Log and eval the first global_step"})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps. \
                                                      By default, it will be set to the num of \
                                                      steps in each epoch, when num_train_epochs do not be overridden"})
    save_steps: int = field(default=5000, metadata={"help": "Save checkpoint every X updates steps. \
                                                   By default, it will be set to the num of \
                                                   steps in each epoch, when num_train_epochs do not be overridden"})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )

    # config the arguments about precision of GPU
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )
    # local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})

    # tpu_num_cores: Optional[int] = field(
    #     default=None, metadata={"help": "TPU: Number of TPU cores (automatically passed by launcher script)"}
    # )
    # tpu_metrics_debug: bool = field(default=False, metadata={"help": "TPU: Whether to print debug metrics"})

    def __post_init__(self):
        pass

    def get_name_abbreviation(self):
        base_result = super().get_name_abbreviation()
        result = {
            "learning_rate": "lr",
            "num_train_epochs": 'ep'
        }
        result.update(base_result)

        return result

@dataclass

class ArgumentsBox:
    model_args: ModelArguments
    data_args: DataArguments
    performing_args: PerformingArguments

    def replace_args(self, replace_dict: Dict[str, Any]):
        if len(replace_dict) <=0:
            return
        replace_dict_ = replace_dict.copy()
        self.model_args = self.model_args.replace_args(replace_dict)
        self.data_args = self.data_args.replace_args(replace_dict)
        self.performing_args = self.performing_args.replace_args(replace_dict)

        if len(replace_dict) != 0:
            raise ValueError

        logger.warning(f"replace some arguments: {str(replace_dict)}")


@dataclass
class ArgumentsTypeBox:
    model_args: type = ModelArguments
    data_args: type = DataArguments
    performing_args: type = PerformingArguments
