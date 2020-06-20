from data import DataSetType, Example, BaseDataSet
from data.corpus import Corpus
import config
from torch.utils.data import Dataset
from typing import List
from argument import DataArguments
from ..corpus import name2corpus_type, DataSetType
from typing import Optional, Dict, Any, Tuple
from torch import Tensor
from ..corpus import ItemsForMetricsComputation
from torch.utils.data import DataLoader
from data import InputFeaturesUpdate, InputFeatures
from dataclasses import replace
import logging

logger = logging.getLogger(__name__)


class DataProxy:
    def __init__(self, data_args: DataArguments, *args, **kwargs):
        self.data_args = data_args
        corpus_name = data_args.task_name
        corpus_obj = name2corpus_type[corpus_name]()
        self.corpus = corpus_obj
        self.input_features_dict = {}
        self.dataset_dict: Dict[DataSetType, BaseDataSet] = {}
        self.dataloader_dict = {}
        self._num_label = None
        self._task_name = None
        self._mute = False
        self.input_feature_class: InputFeatures = kwargs.get('input_feature_class')

    def convert_examples_to_input_features(self, ds_type: DataSetType, *args, **kwargs):
        if ds_type not in self.input_features_dict:
            self._check_mute(kwargs)
            examples = self.corpus.get_examples(ds_type, *args, **kwargs)
            input_features = self._convert_examples_to_features(examples, *args, **kwargs)
            self.input_features_dict[ds_type] = input_features
        return self.input_features_dict[ds_type]

    def _convert_examples_to_features(self, examples: List[Example], *args, **kwargs):
        raise NotImplementedError()

    def get_dataset(self, ds_type: DataSetType, *args, **kwargs) -> Dataset:
        if ds_type not in self.dataset_dict:
            self._check_mute(kwargs)
            dataset = self._create_dataset(ds_type, *args, **kwargs)
            self.dataset_dict[ds_type] = dataset
        return self.dataset_dict[ds_type]

    def _create_dataset(self, ds_type: DataSetType, input_features):
        raise NotImplementedError()

    def get_dataloader(self, ds_type: DataSetType, *args, **kwargs) -> DataLoader:
        if ds_type not in self.dataloader_dict:
            self._check_mute(kwargs)
            dataset = self.get_dataset(ds_type, *args, **kwargs)
            dataloader = self._create_dataloader(dataset, *args, **kwargs)
            self.dataloader_dict[ds_type] = dataloader
        return self.dataloader_dict[ds_type]

    def _create_dataloader(self, dataset: Dataset, *args, **kwargs):
        raise NotImplementedError()

    def _check_mute(self, kwargs: Dict[str, Any]):
        current_setting = kwargs.get('mute')
        if self._mute:
            if isinstance(current_setting, bool) and (not current_setting):
                logger.warning('The global state of mute of data proxy is "on", so can not turn off currently!')
            kwargs['mute'] = True

        # if (not self._mute) and current_setting:
        #     logger.warning('Turn on mute currently!')

    def compute_metrics(self, itmes: ItemsForMetricsComputation):
        return self.corpus.compute_metrics(itmes)

    def merge_datasets(self, ds_types = Tuple[DataSetType]):
        data_sets = []
        for dst in ds_types:
            data_sets.append(self.get_dataset(dst))

        result = BaseDataSet.merge_datasets(tuple(data_sets))
        return result

    def set_datasets(self, ds_type: DataSetType, dataset: BaseDataSet):
        dataset.type = ds_type
        self.dataset_dict[ds_type] = dataset
        self.input_features_dict[ds_type] = dataset.features

    def update_inputfeatures_in_dataset(self, ds_type: DataSetType, updates: List[InputFeaturesUpdate]):
        data_set = self.dataset_dict[ds_type]
        data_set.update_feature(updates)

    @property
    def num_label(self):
        if self._num_label is None:
            self._num_label = self.corpus.get_num_of_labels()
        return self._num_label

    @property
    def task_name(self):
        if self._task_name is None:
            self._task_name = self.corpus.name.lower()
        return self._task_name

    @property
    def mute(self):
        return self._mute

    @mute.setter
    def mute(self, state: bool):
        if state:
            logger.warning('Turn on mute mode for data proxy')
        else:
            logger.warning('Turn off mute mode for data proxy')

        self._mute = state

    def get_num_examples(self, data_type: DataSetType) -> int:
        return len(self.get_dataset(data_type))

    def has_label(self, ds_type: DataSetType):
        return self.corpus.has_label(ds_type=ds_type)

    def is_paraphrase_task(self):
        from data.corpus import name2is_paraphrase
        return name2is_paraphrase[self.task_name]

def _create_data_proxy(proxy_type: type, data_args: DataArguments, *args, **kwargs) -> DataProxy:
    data_proxy = proxy_type(data_args, *args, **kwargs)
    return data_proxy


data_proxy_singleton: Optional[DataProxy] = None


def create_data_proxy(proxy_type: type, data_args: DataArguments, *args, **kwargs) -> DataProxy:
    global data_proxy_singleton
    force = kwargs.get('force') if kwargs.get('force') is not None else False
    if data_proxy_singleton is None or force:
        data_proxy_singleton = _create_data_proxy(proxy_type, data_args, *args, **kwargs)
    return data_proxy_singleton




