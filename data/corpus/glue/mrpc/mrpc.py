from data import OutputMode, DataSetType
import logging
import utils.file_tool as file_tool
from data.corpus import Corpus
from data.corpus._utils import ItemsForMetricsComputation, acc_and_f1
from data import Example, Sentence
from enum import Enum, unique
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from utils.general_tool import is_number
logger = logging.getLogger(__name__)


@unique
class ParapraseLabel(Enum):
    yes = 1
    no = 0


@dataclass(frozen=True)
class MRPCExample(Example):
    sentence_1: Sentence
    sentence_2: Sentence
    label: Optional[ParapraseLabel]
    index: Optional[int]
    def get_texts(self) -> Tuple[Sentence, ...]:
        return self.sentence_1, self.sentence_2


class MRPCorpus(Corpus):
    name = 'MRPC'
    output_mode = OutputMode.classification

    e_id_interval = 10000
    e_id_base_dict = {
        DataSetType.train: 0,
        DataSetType.dev: e_id_interval,
        DataSetType.test: e_id_interval*2
    }
    label_type = ParapraseLabel
    data_path = file_tool.dirname(__file__)
    raw_filename_dict = {
        DataSetType.train: file_tool.connect_path(data_path, 'raw', 'train.tsv'),
        DataSetType.dev: file_tool.connect_path(data_path, 'raw', 'dev.tsv'),
        DataSetType.test: file_tool.connect_path(data_path, 'raw', 'test.tsv')
    }

    def __init__(self):
        super().__init__()
        self.id2sentence = self.id2text

    def get_num_of_labels(self) -> int:
        num = len(ParapraseLabel.__members__)
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
            if i == 0:
                continue
            guid = e_id_base+i

            id_1 = line[1].strip()
            id_2 = line[2].strip()

            if not(is_number(id_1) and is_number(id_2)):
                raise ValueError('The id1 or id2 is not a number at the {}-th line in file:{}'.format(i+1, file_name))

            sent_obj_1 = self._create_text_by_id(id_=id_1, text_type=Sentence, raw=line[3].strip())
            sent_obj_2 = self._create_text_by_id(id_=id_2, text_type=Sentence, raw=line[4].strip())

            label = int(line[0].strip())
            index = None
            #must mask test label
            if ds_type != DataSetType.test:
                if label == ParapraseLabel.yes.value:
                    label = ParapraseLabel.yes
                elif label == ParapraseLabel.no.value:
                    label = ParapraseLabel.no
                else:
                    raise ValueError('The lable error in raw file')
            else:
                index = label
                label = None

            ##if test file have label can try don't mask
            # if ds_type == DataSetType.test:
            #     index = label
            #
            # if label == ParapraseLabel.yes.value:
            #     label = ParapraseLabel.yes
            # elif label == ParapraseLabel.no.value:
            #     label = ParapraseLabel.no
            # else:
            #     raise ValueError('The lable error in raw file')

            e = MRPCExample(id=guid, type=ds_type, sentence_1=sent_obj_1, sentence_2=sent_obj_2,
                            label=label, index=index)
            examples.append(e)

        return examples

    def compute_metrics(self, itmes: ItemsForMetricsComputation):
        return acc_and_f1(itmes.predictions, itmes.label_ids)

