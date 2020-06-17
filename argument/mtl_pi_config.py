from .tfrs_config import TFRsDataArguments, TFRsModelArguments, TFRsPerformingArguments
from dataclasses import dataclass, field
from config import configurator
from framework.mtl.mtl_pi import MTLPIFrameworkProxy
from data.proxy.mtl_pi_proxy import MTLPIDataProxy


@dataclass
class MTLPIModelArguments(TFRsModelArguments):
    pass


@dataclass
class MTLPIDataArguments(TFRsDataArguments):
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class MTLPIPerformingArguments(TFRsPerformingArguments):
    auxiliary_training_epoch: int = field(default=4, metadata={"help": "The training epoch for auxiliary task."})
    auxiliary_learning_rate: float = field(default=4, metadata={"help": "The initial learning rate for auxiliary task."})
    auxiliary_batch_size: int = field(default=8, metadata={"help": "The batch size for auxiliary task."})
    field_name_replace_dict = {
        'num_train_epochs': 'parallel_training_epoch',
        'learning_rate': 'parallel_training_epoch',
        'train_batch_size': 'parallel_train_batch_size'
    }

    def __post_init__(self):
        super().__post_init__()


# @dataclass
# class MTLPIArgumentsBox(ArgumentsBox):
#     model_args: MTLPIModelArguments
#     data_args: MTLPIDataArguments
#     performing_args: MTLPIPerformingArguments

configurator.register_arguments(model_args=MTLPIModelArguments,
                                data_args=MTLPIDataArguments,
                                performing_args=MTLPIPerformingArguments)

configurator.register_data_proxy_type(MTLPIDataProxy)
configurator.register_framework_proxy_type(MTLPIFrameworkProxy)

# class MTLPIConfigurator(Configurator):


