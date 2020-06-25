from .tfrs import TFRsDataArguments, TFRsModelArguments, TFRsPerformingArguments
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MTLPIModelArguments(TFRsModelArguments):
    loss_a: float = field(
        default=0.5,
        metadata={
            "help": "The weight rate for two task"
        },
    )

    def get_name_abbreviation(self):
        base_result = super().get_name_abbreviation()
        result = {
            "loss_a": "mlla",
        }
        result.update(base_result)
        return result


@dataclass(frozen=True)
class MTLPIDataArguments(TFRsDataArguments):
    auxiliary_per_device_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/CPU for training for auxiliary task."}
    )

    def get_name_abbreviation(self):
        base_result = super().get_name_abbreviation()
        result = {
            "auxiliary_per_device_batch_size": "aupdbz",
        }
        result.update(base_result)
        return result


@dataclass(frozen=True)
class MTLPIPerformingArguments(TFRsPerformingArguments):
    auxiliary_training_epoch: int = field(default=6, metadata={"help": "The training epoch for auxiliary task."})
    auxiliary_learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for auxiliary task."})

    skip_revise_predictions: bool = field(default=False, metadata={"help": "The initial learning rate for auxiliary task."})

    # field_name_replace_dict = {
    #     'num_train_epochs': 'parallel_training_epoch',
    #     'learning_rate': 'parallel_training_epoch',
    #     'train_batch_size': 'parallel_train_batch_size'
    # }
    def get_name_abbreviation(self):
        base_result = super().get_name_abbreviation()
        result =  {
            "auxiliary_training_epoch": "auep",
            "auxiliary_learning_rate": "aulr",
        }
        result.update(base_result)
        return result

    def __post_init__(self):
        super().__post_init__()


# @dataclass
# class MTLPIArgumentsBox(ArgumentsBox):
#     model_args: MTLPIModelArguments
#     data_args: MTLPIDataArguments
#     performing_args: MTLPIPerformingArguments


# class MTLPIConfigurator(Configurator):


