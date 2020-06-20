# from data import Example, Sentence
# from enum import Enum,unique
# import torch.utils.data as torch_data
# from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
# from data._utils import collate_batch
# from dataclasses import dataclass
# from typing import Optional, Tuple
#
#
# @unique
# class ParapraseLabel(Enum):
#     yes = 1
#     no = 0
#
#
# @dataclass(frozen=True)
# class MRPCExample(Example):
#     sentence_1: Sentence
#     sentence_2: Sentence
#     label: Optional[ParapraseLabel]
#
#     def get_texts(self) -> Tuple[Sentence, ...]:
#         return self.sentence_1, self.sentence_2
#



