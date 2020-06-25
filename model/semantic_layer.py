import torch
import math
import utils.wmd as wmd
import numpy as np
from enum import Enum, unique


@unique
class SemanticCompareEnum(Enum):
    l2 = 'l2'
    arccos = 'arccos'
    wmd = 'wmd'
    l1 = 'l1'

class SemanticLayer(torch.nn.Module):
    def __init__(self, semantic_compare_func: SemanticCompareEnum):
        self.arsemantic_compare_func = semantic_compare_func
        super().__init__()

    def revise_zero_data(self, tensor):
        boundary = 0.00001 ** 2
        flag_tensor = (tensor < boundary).type(torch.float)
        tensor = tensor + flag_tensor * boundary
        return tensor

    def forward(self, sentence1s, sentence2s, sentence1_lens=None, sentence2_lens=None):
        sentence1s = sentence1s.float()
        sentence2s = sentence2s.float()
        if self.arsemantic_compare_func.value == 'l2':
            result = self.l2distance(sentence1s, sentence2s)

        elif self.arsemantic_compare_func.value == 'arccos':
            result = self.calculate_arc_cos(sentence1s, sentence2s)

        elif self.arsemantic_compare_func.value == 'wmd':
            result = self.word_mover_distance(sentence1s, sentence2s, sentence1_lens, sentence2_lens)

        elif self.arsemantic_compare_func.value == 'l1':
            result = self.l1distance(sentence1s, sentence2s)

        else:
            raise ValueError

        if torch.isnan(result).sum() > 0:
            print(torch.isnan(result))
            raise ValueError
        return result

    def l1distance(self, sentence1s, sentence2s):
        sentence1_len = sentence1s.size()[1]
        sentence2_len = sentence2s.size()[1]
        sentence1s = sentence1s.sum(dim=1) / sentence1_len
        sentence2s = sentence2s.sum(dim=1) / sentence2_len

        result = torch.abs(sentence1s - sentence2s)

        return result

    def l2distance(self, sentence1s, sentence2s):
        sentence1_len = sentence1s.size()[1]
        sentence2_len = sentence2s.size()[1]
        sentence1s = sentence1s.sum(dim=1) / sentence1_len
        sentence2s = sentence2s.sum(dim=1) / sentence2_len

        result = torch.pow(sentence1s - sentence2s, 2)
        if (result == 0).any():
            result = self.revise_zero_data(result)
        result = torch.sqrt(result)

        # result = torch.abs(sentence1s - sentence2s)

        return result

    def calculate_arc_cos(self, input_data1, input_data2):
        shape1 = input_data1.size()
        shape2 = input_data2.size()
        result = []
        for i in range(shape1[0]):
            x = input_data1[i]
            y = input_data2[i]
            result_temp = torch.nn.functional.normalize(x, dim=1).mm(
                torch.nn.functional.normalize(y, dim=1).T).squeeze()
            if torch.isnan(result_temp).sum() > 0:
                print(torch.isnan(result_temp))
                raise ValueError
            # l_length, c_length = result_temp.size()
            index_list = (result_temp >=1 ).nonzero().cpu().numpy()
            index_list = [np.array(index_pair).reshape(-1,1) for index_pair in index_list]
            if len(index_list)>0:
                index_list = np.concatenate(index_list, axis=1)
                result_temp[index_list] -= 1e-6

            index_list = (result_temp <= -1).nonzero().cpu().numpy()
            index_list = [np.array(index_pair).reshape(-1, 1) for index_pair in index_list]
            if len(index_list) > 0:
                index_list = np.concatenate(index_list, axis=1)
                result_temp[index_list] += 1e-6

            # for l in range(l_length):
            #     for c in range(c_length):
            #         if result_temp[l, c] >= 1:
            #             result_temp[l, c] = 1.0 - 1e-8
            #         if result_temp[l, c] <= -1:
            #             result_temp[l, c] = -1.0 + 1e-8
            if (torch.abs(result_temp) >=1 ).any():
                raise ValueError

            result_temp = torch.acos(result_temp) / math.pi
            result_temp = torch.ones_like(result_temp) - result_temp
            result_temp = result_temp.reshape(-1)
            result.append(result_temp)
        result = torch.stack(result, dim=0)

        return result

    def word_mover_distance(self, sentence1s, sentence2s, sentence1_lens, sentence2_lens):
        wmd_obj = wmd.WMD()
        current_batch_size = len(sentence1s)
        result_temp = torch.cdist(sentence1s, sentence2s, p=2)
        result = []
        for batch in range(current_batch_size):
            sentence1 = sentence1s[batch]
            sentence1_np = sentence1.detach().cpu().numpy().copy()[0:sentence1_lens[batch]]
            sentence1_tokens = ["token1_{}".format(i+1) for i in range(sentence1_lens[batch])]

            sentence2 = sentence2s[batch]
            sentence2_np = sentence2.detach().cpu().numpy().copy()[0:sentence2_lens[batch]]
            sentence2_tokens = ["token2_{}".format(i+1) for i in range(sentence2_lens[batch])]

            wvmodel = {sentence1_tokens[i]: sentence1_np[i] for i in range(sentence1_lens[batch])}

            wvmodel.update(
                {sentence2_tokens[i]: sentence2_np[i] for i in range(sentence2_lens[batch])}
            )

            weight_matrix = np.zeros(result_temp[batch].size())
            weight_matrix_temp = wmd_obj.trans_matrix(sentence1_tokens, sentence2_tokens, wvmodel)
            shape = weight_matrix_temp.shape
            weight_matrix[0:shape[0], 0:shape[1]] = weight_matrix_temp
            result.append(result_temp[batch] * torch.from_numpy(weight_matrix).to(dtype=result_temp.dtype,
                                                                                  device=result_temp.device))

        result = torch.stack(result, dim=0)
        result = result.reshape(current_batch_size, -1)
        return result
