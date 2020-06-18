from argument import Arguments, ArgumentsBox, ArgumentsTypeBox, GArgumentParser
from enum import Enum, unique
from dataclasses import fields
from utils.general_tool import singleton
from data.proxy import DataProxy
from framework import FrameworkProxy


# @unique
# class SystemState(Enum):
#     perform = 'perform',
#     tune_hp = 'tuning_hyperparameter'


@singleton
class _Configurator:
    # def __init__(self, args_box_type: type(ArgumentsBox)):
    #     if not issubclass(args_box_type, ArgumentsBox):
    #         raise ValueError("The {} is not the subclass of ArgumentsBox".format(args_box_type))
    #     self.args_box_type = args_box_type
    def __init__(self):
        self.arguments_type_box = None
        self._data_proxy_type = None
        self._framework_proxy_type = None
        self._return_remaining_strings = False

    def _get_arguments_box_when_running(self, args_type_box: ArgumentsTypeBox):
        arguments_name2class = args_type_box.__dict__
        names = arguments_name2class.keys()

        args_parser = GArgumentParser([arguments_name2class[n] for n in names])
        args_obj_tuple = args_parser.parse_args_into_dataclasses(return_remaining_strings=self._return_remaining_strings)

        arg_box_input = {}
        for k, v in zip(names, args_obj_tuple):
            if type(v) != arguments_name2class[k]:
                raise RuntimeError
            arg_box_input[k] = v
        return ArgumentsBox(**arg_box_input)

    def _get_arguments_box_when_tuning_hp(self):
        raise NotImplementedError()

    # def get_arguments_box(self, status: SystemState):
    #     if status == SystemState.perform:
    #         return self._get_arguments_box_when_performing()
    #     elif status == SystemState.perform:
    #         return self._get_arguments_box_when_tuning_hp()
    #     else:
    #         raise ValueError

    def get_arguments_box(self):
        # if not issubclass(args_box_type, ArgumentsBox):
        #     raise ValueError("The {} is not the subclass of ArgumentsBox".format(args_box_type))
        if self.arguments_type_box is None:
            raise RuntimeError('arguments_type_box is None')

        return self._get_arguments_box_when_running(self.arguments_type_box)

    def register_arguments(self, model_args, data_args, performing_args, **kwargs):
        if self.arguments_type_box is None:
            self.arguments_type_box  = ArgumentsTypeBox()
        self.arguments_type_box.model_args = model_args
        self.arguments_type_box.data_args = data_args
        self.arguments_type_box.performing_args = performing_args

        for name, type_ in kwargs:
            if not issubclass(type_, Arguments):
                raise ValueError('The type {} is not the subclass of Arguments'.format(type_))
            self.arguments_type_box.name = type_

    # def register_global_types(self, corpus_type, framework_type, data_proxy_type, framework_manager_type):
    #     self.corpus_type = corpus_type
    #     self.framework_type = framework_type
    #     self.data_proxy_type = data_proxy_type
    #     self.framework_manager_type = framework_manager_type

    @property
    def data_proxy_type(self) -> type:
        return self._data_proxy_type

    @data_proxy_type.setter
    def data_proxy_type(self, data_proxy_type: type):
        if not issubclass(data_proxy_type, DataProxy):
            raise ValueError
        if self._data_proxy_type is not None:
            raise ValueError
        self._data_proxy_type = data_proxy_type

    @property
    def framework_proxy_type(self) -> type:
        return self._framework_proxy_type

    @framework_proxy_type.setter
    def framework_proxy_type(self, framework_proxy_type: type):
        if not issubclass(framework_proxy_type, FrameworkProxy):
            raise ValueError
        if self._framework_proxy_type is not None:
            raise ValueError
        self._framework_proxy_type = framework_proxy_type


configurator = _Configurator()

