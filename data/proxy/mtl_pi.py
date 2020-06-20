from . import DataProxy
from data import DataSetType, InputFeatures, Example
import logging
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch.utils.data as torch_data
from data._utils import collate_batch
from dataclasses import dataclass, replace, fields
from typing import List, Optional, Union
from ..corpus import Corpus
from .tfrs import TFRsDataProxy, TFRsInputFeatures, TFRsDataArguments
from argument.mtl_pi import MTLPIDataArguments
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MTLPIInputFeatures(TFRsInputFeatures):
    auxiliary_label: Optional[Union[int, float]] = None


class MTLPIDataProxy(TFRsDataProxy):
    def __init__(self, data_args: MTLPIDataArguments, *args, **kwargs):
        if data_args.task_name == "elaboration":
            raise ValueError
        if not isinstance(data_args, MTLPIDataArguments):
            raise ValueError("The data_args is not the subclass of {}.".format(MTLPIDataArguments))

        super().__init__(data_args, *args, **kwargs)
        self.data_args = data_args
        # tfs_data_args_field_names = TFRsDataArguments.field_names()

        # next five lines must be fix order
        tfs_data_args_init_kwargs = TFRsDataArguments.search_init_kwargs(self.data_args.names2value)

        primary_data_args = TFRsDataArguments(**tfs_data_args_init_kwargs)

        auxiliary_data_args = replace(primary_data_args, task_name='elaboration',
                                      per_device_train_batch_size=self.data_args.auxiliary_per_device_batch_size )

        self.primary_sub_proxy = TFRsDataProxy(data_args=primary_data_args, input_feature_class = MTLPIInputFeatures)

        self.auxiliary_sub_proxy = TFRsDataProxy(data_args=auxiliary_data_args, input_feature_class = MTLPIInputFeatures)
        # last five lines must be fix order
        pass

    def get_feature_field_for_predict(self, pred):
        return {'auxiliary_label': int(pred)}

    from data import InputFeaturesUpdate

    def revise_invalid_predict_for_primary_task(self, updates: List[InputFeaturesUpdate]):
        i_f_u = updates[0]
        auxiliary_label_key = 'auxiliary_label'
        if auxiliary_label_key not in i_f_u.replace_field_dict:
            raise ValueError
        if not self.primary_sub_proxy.is_paraphrase_task():
            raise ValueError
        if not self.primary_sub_proxy is i_f_u.target_data_proxy:
            raise ValueError
        if i_f_u.dataset_type != DataSetType.train:
            raise ValueError
        if len(updates) != len(self.primary_sub_proxy.get_dataset(DataSetType.train)):
            raise ValueError

        from data.corpus.glue.mrpc.mrpc import ParapraseLabel
        from data.corpus.elaboration.elaboration import Elaboration

        auxiliray_label_old_yes_ids = []
        auxiliray_label_yes_ids = []
        auxiliray_label_no_ids = []
        auxiliray_label_old_no_ids = []
        e_id_set = set(list(self.primary_sub_proxy.corpus.id2example.keys()).copy())
        for i_f_u in updates:
            if i_f_u.replace_field_dict[auxiliary_label_key] == Elaboration.yes.value:
                auxiliray_label_old_yes_ids.append(i_f_u.example_id)
            elif i_f_u.replace_field_dict[auxiliary_label_key] == Elaboration.no.value:
                auxiliray_label_old_no_ids.append(i_f_u.example_id)
            else:
                raise ValueError

            e_id_set.remove(i_f_u.example_id)

            e = self.primary_sub_proxy.corpus.id2example[i_f_u.example_id]

            if e.label == ParapraseLabel.yes:
                i_f_u.replace_field_dict[auxiliary_label_key] = Elaboration.no.value
                auxiliray_label_no_ids.append(e.id)
            elif e.label == ParapraseLabel.no:
                if i_f_u.replace_field_dict[auxiliary_label_key] == Elaboration.yes.value:
                    auxiliray_label_yes_ids.append(e.id)
                else:
                    auxiliray_label_no_ids.append(e.id)
            else:
                raise ValueError

        num_old_yes = len(auxiliray_label_old_yes_ids)
        num_old_no = len(updates) - len(auxiliray_label_old_yes_ids)

        num_yes = len(auxiliray_label_yes_ids)
        num_no = len(updates) - len(auxiliray_label_yes_ids)
        logger.info(f"The original number of auxiliray yes label predicted  is: {num_old_yes}")
        logger.info(f"The original number of auxiliray no label predicted is: {num_old_no}")
        logger.info(f"After revise, the number of auxiliray yes label is: {num_yes}")
        logger.info(f"After revise, the number of auxiliray no label is: {num_no}")

        if num_old_no + num_old_yes != len(updates):
            raise ValueError

        if num_old_no != len(updates) - num_yes:
            raise ValueError

        if len(e_id_set) !=0:
            raise ValueError

        check_yes = set()
        check_no = set()
        for i_f_u in updates:
            if i_f_u.replace_field_dict[auxiliary_label_key] == Elaboration.yes.value:
                check_yes.add(i_f_u.example_id)
            else:
                check_no.add(i_f_u.example_id)

        if (len(check_yes) != len(auxiliray_label_yes_ids)) or (len(check_no) != len(auxiliray_label_no_ids)):
            raise ValueError

        detail_info = {
            'The original number of auxiliray-yes-label predicted', num_old_yes,
            'The original number of auxiliray-no-label predicted', num_old_no,
            'After revise, the number of auxiliray-yes-label', num_yes,
            'After revise, the number of auxiliray-no-label', num_no,
        }
        return updates, detail_info


