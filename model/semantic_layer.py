import torch
import math
import utils.wmd as wmd
import numpy as np
from enum import Enum, unique


@unique
class DistanceTypeEnum(Enum):
    dim = 'dim'
    dim_l1 = 'dim_l1'


class SemanticLayer(torch.nn.Module):
    def __init__(self, distance_type_name: str):
        self.distance_type_name = distance_type_name
        super().__init__()

    def forward(self, sentence1s, sentence2s, sentence1_lens=None, sentence2_lens=None):
        sentence1s = sentence1s.float()
        sentence2s = sentence2s.float()

        if len(sentence1s.size()) != 3:
            raise ValueError

        if sentence1_lens is None:
            sentence1_lens = sentence1s.size()[1]
            sentence2_lens = sentence2s.size()[1]

        if self.distance_type_name == 'dim':
            result = self.dim_distance(sentence1s, sentence2s, sentence1_lens, sentence2_lens)

        elif self.distance_type_name == 'dim_l1':
            dim = self.dim_distance(sentence1s, sentence2s, sentence1_lens, sentence2_lens)
            # l1 = self.l1_distance(sentence1s, sentence2s, sentence1_lens, sentence2_lens)
            # l1 = l1.unsqueeze(dim=1)
            l1 = dim.sum(dim=1, keepdim=True)

            result = torch.cat([dim, l1], dim=1)

        else:
            raise ValueError

        if torch.isnan(result).sum() > 0:
            print(torch.isnan(result))
            raise ValueError
        return result

    def dim_distance(self, sentence1s, sentence2s, sentence1_lens, sentence2_lens):
        sentence1s = sentence1s.sum(dim=1) / sentence1_lens
        sentence2s = sentence2s.sum(dim=1) / sentence2_lens

        dim = torch.abs(sentence1s - sentence2s)

        if len(dim.size()) != 2:
            raise ValueError

        return dim

    def l1_distance(self, sentence1s, sentence2s, sentence1_lens, sentence2_lens):
        sentence1s = sentence1s.sum(dim=1) / sentence1_lens
        sentence2s = sentence2s.sum(dim=1) / sentence2_lens

        result = (sentence1s-sentence2s).abs().sum(dim=1)

        if len(result.size()) != 1:
            raise ValueError

        return result
