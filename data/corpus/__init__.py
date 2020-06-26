from data import DataSetType, Example, Text
from typing import Tuple, List, Dict, Any
import torch
from data.corpus._utils import ItemsForMetricsComputation
import logging
logger = logging.getLogger(__name__)


class Corpus:
    dataset_types = (DataSetType.train, DataSetType.dev, DataSetType.test)
    name:str = None
    output_mode = None

    e_id_interval = 10000
    e_id_base_dict = {
        DataSetType.train: 0,
        DataSetType.dev: e_id_interval,
        DataSetType.test: e_id_interval*2
    }
    raw_filename_dict = None

    label_type = None

    @property
    def value2name_of_label_type(self):
        result = {}
        for l_t in self.label_type:
            if l_t.value in result:
                raise ValueError
            result[l_t.value] = l_t.name
        return result

    def __init__(self):
        self.type2examples = {}
        self.id2example: Dict[int, Example] = {}
        self.id2text = {}

    def get_examples(self, ds_type: DataSetType, *args, **kwargs):
        if not isinstance(ds_type, DataSetType):
            raise ValueError('The data set type is not a invalid, do not come from DataSetType')

        if ds_type not in self.dataset_types:
            raise ValueError('The data set type is not belong to the {} corpus'.format(self.name))

        if ds_type not in self.type2examples:
            logger.info('Load {} {} set.'.format(self.name, ds_type.name))
            examples = self._get_examples(ds_type, *args, **kwargs)
            logger.info('Load {} examples from file: {}.'.format(len(examples), self.raw_filename_dict[ds_type]))

            self.type2examples[ds_type] = examples
            for e in examples:
                if str(e.id) in self.id2example:
                    raise ValueError
                self.id2example[int(e.id)] = e

        return self.type2examples[ds_type]

    def get_an_example(self, ds_type: DataSetType, *args, **kwargs):
        examples = self.get_examples(ds_type, *args, **kwargs)
        if len(examples) <= 0:
            raise ValueError
        return examples[0]

    def has_label(self, ds_type: DataSetType, *args, **kwargs):
        example = self.get_an_example(ds_type=ds_type, *args, **kwargs)
        if hasattr(example, 'label'):
            if isinstance(example.label, self.label_type):
                return True
            else:
                raise ValueError
        else:
            return False

    def _get_examples(self, ds_type: DataSetType, *args, **kwargs) -> Tuple[Example, ...]:
        raise NotImplementedError()

    def get_num_of_labels(self):
        raise NotImplementedError()

    def compute_metrics(self, itmes: ItemsForMetricsComputation):
        if itmes.predictions.size() is not itmes.label_ids.size():
            raise RuntimeError
        raise NotImplementedError()

    def _create_text_by_id(self, id_: str, text_type: type = Text, **init_args) -> Any:

        if id_ not in self.id2text:
            text_obj = text_type(id=id_, **init_args)
            self.id2text[id_] = text_obj
        else:
            text_obj = self.id2text[id_]

        return text_obj

    def _save_examples(self, e_ids: Tuple[int, ...], file_name):
        save_data = [f'number of examples: {len(e_ids)}']
        for id_ in e_ids:
            example = self.id2example[id_]
            texts = example.get_texts()
            save_data.append(f'{id_}')
            save_data.extend([str(t.raw) for t in texts])
            save_data.append('')

        from utils import file_tool
        file_tool.makedir(file_tool.dirname(file_name))
        file_tool.save_list_data(save_data, file_name, 'w')

    def _e_ids_and_filename_tuples_according_to_e_id2predictions(self, ds_type: DataSetType, e_id2predictions: Dict[int, Any]):
        raise NotImplementedError()

    def save_examples_according_to_e_id2predictions(self, ds_type: DataSetType, e_id2predictions: Dict[int, Any]):
        e_ids_and_filename_tuples = self._e_ids_and_filename_tuples_according_to_e_id2predictions(ds_type, e_id2predictions)
        for e_ids, filename in e_ids_and_filename_tuples:
            self._save_examples(e_ids, filename)

    pass


# import data module
from .glue.qqp import qqp
from .glue.mrpc.mrpc import MRPCorpus
from .discourse.elaboration.elaboration import ElabCorpus
from .discourse.coherence.coherence import CoherenceCorpus
from .discourse.resemblance.resemblance import ResemblanceCorpus

from ._utils import ItemsForMetricsComputation

name2corpus_type = {
    'mrpc': MRPCorpus,
    'elaboration': ElabCorpus,
    'coherence': CoherenceCorpus,
    'resemblance': ResemblanceCorpus
}

name2is_paraphrase = {
    'mrpc': True,
    'elaboration': False
}