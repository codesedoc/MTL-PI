from ._argument import Arguments, ArgumentsBox, ArgumentsTypeBox, ModelArguments, PerformingArguments, DataArguments
from ._utils import GArgumentParser

from enum import Enum


class ArgumentsType(Enum):
    model = 'model'
    data = 'data'
    perform = 'perform'
