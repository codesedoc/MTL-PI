from ._argument import DataArguments, ModelArguments, PerformingArguments
from dataclasses import dataclass, field
from typing import Optional
from config import configurator


@dataclass
class TFRsModelArguments(ModelArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )


@dataclass
class TFRsDataArguments(DataArguments):
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class TFRsPerformingArguments(PerformingArguments):
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})


# class TFRsConfigurator(Configurator):
#     def _get_arguments_box_when_tuning_hp(self):
#         pass
