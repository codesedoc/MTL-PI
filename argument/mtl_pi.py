from .tfrs import TFRsDataArguments, TFRsModelArguments, TFRsPerformingArguments
from dataclasses import dataclass, field


@dataclass
class MTLPIModelArguments(TFRsModelArguments):
    pass


@dataclass
class MTLPIDataArguments(TFRsDataArguments):
    auxiliary_per_device_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/CPU for training for auxiliary task."}
    )

@dataclass
class MTLPIPerformingArguments(TFRsPerformingArguments):
    auxiliary_training_epoch: int = field(default=6, metadata={"help": "The training epoch for auxiliary task."})
    auxiliary_learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for auxiliary task."})


    # field_name_replace_dict = {
    #     'num_train_epochs': 'parallel_training_epoch',
    #     'learning_rate': 'parallel_training_epoch',
    #     'train_batch_size': 'parallel_train_batch_size'
    # }

    def __post_init__(self):
        super().__post_init__()


# @dataclass
# class MTLPIArgumentsBox(ArgumentsBox):
#     model_args: MTLPIModelArguments
#     data_args: MTLPIDataArguments
#     performing_args: MTLPIPerformingArguments


# class MTLPIConfigurator(Configurator):


