from data import OutputMode, DataSetType
import logging
import utils.file_tool as file_tool
from data.corpus import Corpus
from data.corpus._utils import ItemsForMetricsComputation, acc_and_f1
from data import Example, Text
from enum import Enum, unique
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from utils.general_tool import is_number
logger = logging.getLogger(__name__)


@unique
class Elaboration(Enum):
    yes = 1
    no = 0


@dataclass(frozen=True)
class ElabExample(Example):
    segment_1: Text
    segment_2: Text
    label: Optional[Elaboration]
    index: Optional[int]
    original_id: int

    def get_texts(self) -> Tuple[Text, Text]:
        return self.segment_1, self.segment_2


class ElabCorpus(Corpus):
    name = 'Elaboration'
    output_mode = OutputMode.classification

    e_id_interval = 10000
    e_id_base_dict = {
        DataSetType.train: 0,
        DataSetType.dev: e_id_interval,
        DataSetType.test: e_id_interval*2
    }
    label_type = Elaboration
    data_path = file_tool.dirname(__file__)
    raw_filename_dict = {
        DataSetType.train: file_tool.connect_path(data_path, 'raw', 'train_set.txt'),
        DataSetType.dev: file_tool.connect_path(data_path, 'raw', 'dev_set.txt'),
        DataSetType.test: file_tool.connect_path(data_path, 'raw', 'test_set.txt')
    }

    def __init__(self):
        super().__init__()
        self.id2segment = self.id2text

    def get_num_of_labels(self) -> int:
        num = len(Elaboration.__members__)
        return num

    def _get_examples(self, ds_type: DataSetType, *args, **kwargs):
        """Creates examples for the training and dev sets."""
        e_id_base = self.e_id_base_dict[ds_type]
        examples = []

        file_name = self.raw_filename_dict[ds_type]
        lines = file_tool.read_tsv(file_name)

        if len(lines) > self.e_id_interval:
            raise ValueError

        for (i, line) in enumerate(lines):
            if len(line) != 7:
                raise ValueError

            if i == 0:
                continue
            guid = e_id_base+i
            o_id = int(line[0].strip())
            id_1 = line[2].strip()
            id_2 = line[4].strip()

            if not(is_number(id_1) and is_number(id_2)):
                raise ValueError('The id1 or id2 is not a number at the {}-th line in file:{}'.format(i+1, file_name))

            segment_obj_1 = self._create_text_by_id(id_=id_1, text_type=Text, raw=line[3].strip())
            segment_obj_2 = self._create_text_by_id(id_=id_2, text_type=Text, raw=line[5].strip())

            label = int(line[1].strip())

            if label == Elaboration.yes.value:
                label = Elaboration.yes
            elif label == Elaboration.no.value:
                label = Elaboration.no
            else:
                raise ValueError('The lable error in raw file')

            index = None
            if ds_type == DataSetType.test:
                index = i

            e = ElabExample(id=guid, type=ds_type, segment_1=segment_obj_1, segment_2=segment_obj_2,
                            label=label, index=index, original_id=o_id)
            examples.append(e)

        return examples

    def compute_metrics(self, itmes: ItemsForMetricsComputation):
        return acc_and_f1(itmes.predictions, itmes.label_ids)

