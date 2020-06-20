from . import DataProxy
from argument.tfrs import TFRsDataArguments
from data import DataSetType, InputFeatures, Example
import logging
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch.utils.data as torch_data
from data._utils import collate_batch
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
import transformers

from ..corpus import Corpus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TFRsInputFeatures(InputFeatures):
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    index: Optional[int] = None

from .. import BaseDataSet

class TFRsDataSet(BaseDataSet):
    pass


class TFRsDataProxy(DataProxy):
    def __init__(self, data_args: TFRsDataArguments, *args, **kwargs):
        if not isinstance(data_args, TFRsDataArguments):
            raise ValueError("The data_args is not the subclass of {}.".format(TFRsDataArguments))

        super().__init__(data_args, *args, **kwargs)
        self.data_args = data_args
        self.tokenizer: Optional[transformers.PreTrainedTokenizer] = None

    def _convert_examples_to_features(self, examples: List[Example], *args, **kwargs):
        InputFeature_class = TFRsInputFeatures
        if self.input_feature_class is not None:
            InputFeature_class = self.input_feature_class

        tokenizer = self.tokenizer
        max_length = None
        if 'max_length_of_input_tokens' in kwargs:
            max_length = kwargs['max_length_of_input_tokens']
        if max_length is None:
            max_length = tokenizer.max_len
        dataset_name = self.corpus.name
        output_mode = self.corpus.output_mode
        mute = True if kwargs.get('mute') is not None else False

        if not mute:
            logger.info("Using label type %s for data set %s" % (self.corpus.label_type.__name__, dataset_name))
            logger.info("Using output mode %s for data set %s" % (output_mode, dataset_name))

        def label_from_example(example: Example) -> Union[int, float, None]:
            if (not hasattr(example, 'label')) or (example.label is None):
                return None
            if not isinstance(example.label, self.corpus.label_type):
                raise ValueError

            from data import OutputMode
            if output_mode == OutputMode.classification:
                return example.label.value
            elif output_mode == OutputMode.regression:
                return float(example.label)
            else:
                raise KeyError(output_mode)

        labels = [label_from_example(example) for example in examples]
        batch_encoding = tokenizer.batch_encode_plus(
            [self.get_raw_texts_for_input(example) for example in examples], max_length=max_length, pad_to_max_length=True,
        )

        features = []
        tokens_list = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}

            feature = InputFeature_class(**inputs, example_id=examples[i].id, label=labels[i], index=examples[i].index)
            features.append(feature)
            tokens_list.append(tokenizer.convert_ids_to_tokens(inputs['input_ids']))

        if not mute:
            for i, example in enumerate(examples[:3]):
                logger.info("*** Example ***")
                logger.info("example_id: %s" % example.id)
                logger.info("features: %s" % features[i])
                logger.info("tokens: %s" % tokens_list[i])
        return features
        pass

    def _create_dataset(self, ds_type: DataSetType, *args, **kwargs):
        input_features = self.convert_examples_to_input_features(ds_type, *args, max_length_of_input_tokens=self.data_args.max_seq_length, **kwargs)
        return TFRsDataSet(ds_type, input_features)

    def _create_dataloader(self, dataset:TFRsDataSet, *args, **kwargs):
        if dataset is None:
            raise ValueError("Requires a dataset.")

        if dataset.shuffle:
            sampler = RandomSampler(dataset)
            batch_size = self.data_args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = self.data_args.eval_batch_size

        data_loader = torch_data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_batch,
            drop_last=self.data_args.dataloader_drop_last,
        )

        mute = kwargs.get('mute', False)
        if not mute:
            logger.info("Create %s data loader" % str(dataset.type))

        return data_loader

    def get_raw_texts_for_input(self, example: Example) -> Union[Tuple[str, str], str]:
        texts = example.get_texts()
        if len(texts) >2 or len(texts)<=0:
            raise ValueError
        if len(texts) == 1:
            return texts[0].raw
        else:
            return texts[0].raw, texts[1].raw