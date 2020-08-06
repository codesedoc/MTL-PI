from .tfrs import TFRsDataArguments, TFRsModelArguments, TFRsPerformingArguments
from dataclasses import dataclass, field
from enum import Enum, unique
from model import DistanceTypeEnum


@unique
class FeatureComparedEnum(Enum):
    cls = 'cls'
    tokens = 'tokens'


@unique
class ModeForTrainData(Enum):
    delay = 'delay'
    cross = 'cross'
    mix = 'mix'


@dataclass(frozen=True)
class MTLPIModelArguments(TFRsModelArguments):
    # loss_a: float = field(
    #     default=0.5,
    #     metadata={
    #         "help": "The weight rate for two task"
    #     },
    # )

    # distance_type: str = field(
    #     default=DistanceTypeEnum.dim.value, metadata={"help": "whether combine two texts as input when pass to transformer"}
    # )
    #
    # feature_compared: str = field(
    #     default=FeatureComparedEnum.cls.value, metadata={"help": "Which feature is used to compare"}
    # )

    adjust_prediction: bool = field(default=False, metadata={"help": "Adjust the primary prediction with the auxiliary information."})

    single_task: bool = field(default=False, metadata={"help": "Chose single task, equal to the baseline."})

    tune_off_auxiliary_when_parallel: bool = field(default=False, metadata={"help": "tune_off_auxiliary_when_parallel."})

    calibrator_weight: float = field(default=1.0, metadata={"help": "The weight of calibrator."})

    learn_calibrator_weight: bool = field(default=False, metadata={"help": "Whether learn weight by model itself."})

    def get_name_abbreviation(self):
        base_result = super().get_name_abbreviation()
        result = {
            "loss_a": "mlla",
            'calibrator_weight': 'c_weight'
        }
        result.update(base_result)
        return result

    def __post_init__(self):
        # if not self.split_two_texts_as_input and (self.feature_compared == FeatureComparedEnum.tokens.value):
        #     raise ValueError

        if self.single_task and self.adjust_prediction:
            raise ValueError


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
    auxiliary_training_epoch: int = field(default=2, metadata={"help": "The training epoch for auxiliary task."})
    auxiliary_learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for auxiliary task."})

    train_mode: ModeForTrainData = field(default=ModeForTrainData.delay,
                                          metadata={"help": "The mode for train data"})

    # skip_revise_predictions: bool = field(default=False, metadata={"help": "The initial learning rate for auxiliary task."})

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

    def to_json_string(self):
        return str(self)


# @dataclass
# class MTLPIArgumentsBox(ArgumentsBox):
#     model_args: MTLPIModelArguments
#     data_args: MTLPIDataArguments
#     performing_args: MTLPIPerformingArguments


# class MTLPIConfigurator(Configurator):


