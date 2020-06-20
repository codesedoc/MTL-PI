from enum import Enum, unique
import torch.utils.data as torch_data
# from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
import dataclasses
from dataclasses import dataclass, field, replace
import json
from typing import List, Optional, Union, Any, Tuple, Dict


# define some unify enumerate
@unique
class DataSetType(Enum):
    train = 'train'
    dev = 'dev'
    test = 'test'


# define data structures
@dataclass(frozen=True)
class Text:
    id: int
    raw: str
    pass


@dataclass(frozen=True)
class Sentence(Text):
    # words: List[str]
    pass


@dataclass(frozen=True)
class Example:
    id: int
    type: DataSetType
    label: Union[Enum, float]
    index: Any
    pass

    def get_texts(self) -> Tuple[Text, ...]:
        raise NotImplementedError()


@dataclass(frozen=True)
class InputFeatures:
    example_id: int
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"





@dataclass(frozen=True)
class InputFeaturesUpdate:
    target_data_proxy: Any
    dataset_type: DataSetType
    example_id: int
    replace_field_dict: Dict[str, Any]


@unique
class OutputMode(Enum):
    classification = "classification"
    regression = "regression"


from typing import Tuple, Union
from torch.utils.data import Dataset


class BaseDataSet(Dataset):
    def __init__(self, type_: Union[DataSetType, Tuple[DataSetType, ...]], features: List[InputFeatures]):
        self._type = type_
        self._last_type = None
        if type_ == DataSetType.train:
            self._shuffle = True
        else:
            self._shuffle = False

        self._features = features
        self._e_id2f_index = {int(f.example_id): i for i, f in enumerate(features)}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    @property
    def features(self):
        return tuple(self._features)

    @features.setter
    def features(self, new_features):
        raise ValueError

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, flag: bool):
        if not isinstance(flag, bool):
            raise ValueError
        self._shuffle = flag

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type_: DataSetType):
        if not isinstance(type_, DataSetType):
            raise ValueError
        if type_ != DataSetType.test:
            self._replace_fields_for_features(range(len(self._features)), {'index': -1})

        if type_ == DataSetType.train:
            self._shuffle = True
        else:
            self._shuffle = False

        self._last_type = self._type
        self._type = type_

    def deleted_features(self):
        self._features.clear()

    def _replace_fields_for_features(self, f_indexes, replace_dict):
        for index in f_indexes:
            self._features[index] = replace(self._features[index], **replace_dict)

    def update_feature(self, update_objs: List[InputFeaturesUpdate]):
        for uo in update_objs:
            index = self._e_id2f_index[int(uo.example_id)]
            feature = self._features[index]
            new_feature = replace(feature, **uo.replace_field_dict)
            self._features[index] = new_feature

    @staticmethod
    def merge_datasets(data_sets: Tuple[Dataset, ...]):
        type_ = []
        features = []
        for ds in data_sets:
            if not isinstance(ds, BaseDataSet):
                raise ValueError
            type_.append(ds.type)
            features.extend(list(ds.features))
            ds.deleted_features()

        return BaseDataSet(type_=tuple(type_), features=features)

