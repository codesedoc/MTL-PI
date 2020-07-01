from data import OutputMode, DataSetType
import logging
import utils.file_tool as file_tool
from data.corpus import Corpus
from data.corpus._utils import ItemsForMetricsComputation, acc_and_f1
from data import Example, Sentence
from enum import Enum, unique
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from utils.general_tool import is_number
from utils import file_tool
logger = logging.getLogger(__name__)


@unique
class ParapraseLabel(Enum):
    yes = 1
    no = 0


@dataclass(frozen=True)
class QQPExample(Example):
    question_1: Sentence
    question_2: Sentence
    label: Optional[ParapraseLabel]
    index: Optional[int]

    def get_texts(self) -> Tuple[Sentence, ...]:
        return self.question_1, self.question_2


class QQPCorpus(Corpus):
    name = 'QQP'
    output_mode = OutputMode.classification

    #set the use_rate to choose the part we utilized
    use_rate = 0.01

    e_id_interval = 1e+6
    e_id_base_dict = {
        DataSetType.train: 0,
        DataSetType.dev: e_id_interval,
        DataSetType.test: e_id_interval*2
    }
    label_type = ParapraseLabel
    data_path = file_tool.dirname(__file__)
    raw_path = file_tool.connect_path(data_path, 'raw')
    raw_filename_dict = {
        DataSetType.train: file_tool.connect_path(raw_path, 'train.tsv'),
        DataSetType.dev: file_tool.connect_path(raw_path, 'dev.tsv'),
        DataSetType.test: file_tool.connect_path(raw_path, 'test.tsv')
    }

    examples_filename_dict = {
        DataSetType.train: file_tool.connect_path(raw_path, 'pkl', 'train_examples.pkl'),
        DataSetType.dev: file_tool.connect_path(raw_path, 'pkl', 'dev_examples.pkl'),
        DataSetType.test: file_tool.connect_path(raw_path, 'pkl', 'test_examples.pkl')
    }

    def __init__(self):
        super().__init__()
        self.id2question = self.id2text

    def get_num_of_labels(self) -> int:
        num = len(ParapraseLabel.__members__)
        return num

    def _get_examples(self, ds_type: DataSetType, *args, **kwargs):
        """Creates examples for the training and dev sets."""
        def load_example_from_file():
            e_id_base = self.e_id_base_dict[ds_type]
            examples = []
            file_name = self.raw_filename_dict[ds_type]
            lines = file_tool.read_tsv(file_name)
            text2id = {}
            if len(lines) > self.e_id_interval:
                raise ValueError
            invalid_line = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue

                guid = e_id_base+i

                ## must mask test label
                if ds_type != DataSetType.test:
                    if len(line) != 6:
                        invalid_line.append(i)
                        continue

                    id_1 = line[1].strip()
                    id_2 = line[2].strip()

                    if not(is_number(id_1) and is_number(id_2)):
                        raise ValueError('The id1 or id2 is not a number at the {}-th line in file:{}'.format(i+1, file_name))

                    sent_obj_1 = self._create_text_by_id(id_=id_1, text_type=Sentence, raw=line[3].strip())
                    sent_obj_2 = self._create_text_by_id(id_=id_2, text_type=Sentence, raw=line[4].strip())

                    label = int(line[-1].strip())
                    index = None

                    if label == ParapraseLabel.yes.value:
                        label = ParapraseLabel.yes
                    elif label == ParapraseLabel.no.value:
                        label = ParapraseLabel.no
                    else:
                        raise ValueError('The lable error in raw file')
                else:
                    if len(line) != 3:
                        raise ValueError

                    if line[1].strip() not in text2id:
                        text2id[line[1].strip()] = len(text2id)

                    if line[2].strip() not in text2id:
                        text2id[line[2].strip()] = len(text2id)

                    q_id1 = text2id[line[1].strip()]
                    q_id2 = text2id[line[2].strip()]

                    sent_obj_1 = self._create_text_by_id(id_=q_id1, text_type=Sentence, raw=line[1].strip())
                    sent_obj_2 = self._create_text_by_id(id_=q_id2, text_type=Sentence, raw=line[2].strip())

                    index = int(line[0].strip())
                    label = None

                e = QQPExample(id=guid, type=ds_type, question_1=sent_obj_1, question_2=sent_obj_2,
                               label=label, index=index)

                examples.append(e)

            if len(invalid_line) != 0:
                logger.warning(f"The total line of {ds_type} data file is {len(lines)}, but {len(invalid_line)} lines are invalid")
            logger.info(f"Load {len(examples)} examples and {len(self.id2question)} questrions from {file_name}")

            return examples

        pkl_file_name = self.examples_filename_dict[ds_type]

        if file_tool.is_file(pkl_file_name):
            examples = file_tool.load_data_pickle(pkl_file_name)
            logger.info(f"Load {len(examples)} examples from {pkl_file_name}")
        else:
            examples = load_example_from_file()
            file_tool.makedir(file_tool.dirname(pkl_file_name))
            file_tool.save_data_pickle(examples, pkl_file_name)
            logger.info(f"Save {len(examples)} examples to {pkl_file_name}")

        # if ds_type == DataSetType.test:
        #     examples = examples[: round(len(examples)*self.use_rate)]
        # else:
        #     examples = self._get_part_of_data(examples)

        return examples

    def compute_metrics(self, itmes: ItemsForMetricsComputation):
        return acc_and_f1(itmes.predictions, itmes.label_ids)

    def _get_part_of_data(self, examples: List[QQPExample]):
        def split_examples(e_s):
            p_e = []
            non_p_e = []
            for e in e_s:
                if e.label == ParapraseLabel.yes:
                    p_e.append(e)
                elif e.label == ParapraseLabel.no:
                    non_p_e.append(e)
                else:
                    raise ValueError
            num_example = len(e_s)
            example_rate = f'{round(len(p_e)/ num_example, 2)} : {round(len(non_p_e)/ num_example, 2)}'
            logger.info(f"Example rate (para({len(p_e)}) : non_para{len(non_p_e)}) -- {example_rate},)")

            return p_e, non_p_e

        para_e, non_para_e = split_examples(examples.copy())

        new_exampls = []
        new_para_es = self._sample_from_example_list(para_e, self.use_rate)
        new_non_para_es = self._sample_from_example_list(non_para_e, self.use_rate)

        new_exampls.extend(new_para_es)
        new_exampls.extend(new_non_para_es)

        logger.info("Check result of sample")
        check_para_es, check_non_para_es = split_examples(new_exampls.copy())

        if (len(check_para_es) != len(new_para_es)) or (len(check_non_para_es) != len(new_non_para_es)):
            raise ValueError

        logger.info(f"Total number of examples is : {len(examples)} . Now use {len(new_exampls)} examples by the rate {self.use_rate} !")

        return new_exampls

    def _sample_from_example_list(self, org_list, sample_rate):
        import random
        org_num = len(org_list)
        sample_indexes = set()
        org_indexes = list(range(org_num))
        while (len(sample_indexes) < round(org_num * sample_rate)):
            index = org_indexes.pop(random.randint(0, len(org_indexes)-1))
            sample_indexes.add(index)

        samples = []
        example_list = org_list.copy()
        for index in sample_indexes:
            samples.append(example_list[index])

        org_examples_id_set = set()
        samples_id_set = set()

        for e_id in samples_id_set:
            if e_id not in org_examples_id_set:
                raise ValueError

        return samples

    def _e_ids_and_filename_tuples_according_to_e_id2predictions(self, ds_type: DataSetType, e_id2predictions: Dict[int, Any], output_dir = None):
        TP_e_ids = []
        TN_e_ids = []
        FP_e_ids = []
        FN_e_ids = []
        import numpy as np
        from utils.general_tool import is_number
        for e_id, pred in e_id2predictions.items():
            if isinstance(pred, ParapraseLabel):
                pred = pred.value

            elif not isinstance(pred, int):
                raise ValueError

            example = self.id2example[e_id]
            if example.label == ParapraseLabel.yes:
                if pred == ParapraseLabel.yes.value:
                    TP_e_ids.append(e_id)
                else:
                    FN_e_ids.append(e_id)
            else:
                if pred == ParapraseLabel.no.value:
                    TN_e_ids.append(e_id)
                else:
                    FP_e_ids.append(e_id)

        if output_dir is None:
            output_path = file_tool.connect_path(self.data_path, 'predict_output', ds_type.value)
        else:
            output_path = file_tool.connect_path(output_dir, 'examples_output', ds_type.value)

        return (
            (TP_e_ids, file_tool.connect_path(output_path, 'TP.txt')),
            (TN_e_ids, file_tool.connect_path(output_path, 'TN.txt')),
            (FP_e_ids, file_tool.connect_path(output_path, 'FP.txt')),
            (FN_e_ids, file_tool.connect_path(output_path, 'FN.txt'))
        )