import torch.utils.data as torch_data
# import corpus.base_corpus as base_corpus
import random
import numpy as np
import torch


class MyDateSet(torch_data.Dataset):
    def __init__(self, examples):
        super().__init__()
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]
        result_dict = {
            'example_id': np.array([int(example.id)], dtype=np.int),
            'label': np.array([int(example.label)], dtype=np.int),
            # 'sentence1_id': torch.tensor(example_dict['sentence1_id']).to(dtype=torch.int),
            # 'sentence1': tuple(example_dict['sentence1']),
            # 'syntax_info1': self.dependencies2adj_matrix(example_dict['syntax_info1']).to(dtype=torch.float32),
            # 'sentence1_len': torch.tensor([example_dict['sentence1_len']]).to(dtype=torch.int),
            #
            # 'sentence2_id': torch.tensor(example_dict['sentence2_id']).to(dtype=torch.int),
            # 'sentence2': tuple(example_dict['sentence2']),
            # 'syntax_info2': self.dependencies2adj_matrix(example_dict['syntax_info2']).to(dtype=torch.float32),
            # 'sentence2_len': torch.tensor([example_dict['sentence2_len']]).to(dtype=torch.int),
            # # 'sentence_pair_tokens': torch.tensor([example_dict['sentence_pair_tokens']]).to(dtype=torch.int),
            # # 'segment_ids': torch.tensor([example_dict['segment_ids']]).to(dtype=torch.int),
            # # 'SEP_index': torch.tensor([example_dict['SEP_index']]).to(dtype=torch.int),

        }
        return result_dict


class MyDataLoader(torch_data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.example_dict = None


class ExampleManager:
    def __init__(self, examples, example_type="2class"):
        self.examples = examples
        if example_type == "2class":
            self.positives = []
            self.negatives = []
            self.__classification__()

    def __classification__(self):
        self.positives.clear()
        self.negatives.clear()
        for example in self.examples:
            label = int(example.label)
            if label == 1:
                self.positives.append(example)
            elif label == 0:
                self.negatives.append(example)
            else:
                raise ValueError

    def count_of_examples(self):
        return len(self.examples)

    def ratio_of_labels(self):
        return len(self.positives), len(self.negatives), len(self.positives)/len(self.negatives)

    def divide_part_of_classification(self, classification_dicts, k_part):
        count = len(classification_dicts)
        result = [[] for k in range(k_part)]
        for i in range(count):
            result[i % k_part].append(classification_dicts[i])
        for i in range(k_part):
            if len(result[i]) == 0:
                raise ValueError
        return result

    def divide_into_groups_keep_rate(self, k_group):
        positive_parts = self.divide_part_of_classification(self.positives, k_group)
        negative_parts = self.divide_part_of_classification(self.negatives, k_group)
        if len(positive_parts) != len(negative_parts):
            raise ValueError
        result = [[] for k in range(k_group)]
        # shuffle_indexes = [k for k in range(k_group)]
        # shuffle_indexes = random.shuffle(shuffle_indexes)

        for i, part_pair in enumerate(zip(positive_parts, negative_parts)):
            positive_part, negative_part = part_pair
            result[i] = positive_part.copy()
            result[i].extend(negative_part)
            # print([int(e.label) for e in result[i]])
            random.shuffle(result[i])
            # print([int(e.label) for e in result[i]])
        return result


class DataManager:
    def __init__(self, corpus_obj):
        super().__init__()
        self.corpus = corpus_obj
        self.train_example_manager = ExampleManager(corpus_obj.train_example_list, corpus_obj.data_type)
        self.test_example_manager = ExampleManager(corpus_obj.test_example_list, corpus_obj.data_type)
        if hasattr(corpus_obj, "dev_example_list"):
            self.dev_example_manager = ExampleManager(corpus_obj.dev_example_list, corpus_obj.data_type)

        self.loader_dict = None
    # def __setattr__(self, key, value):
    #     super().__setattr__(key, value)
    #     print('set {} as {}'.format(key, value))
    example_groups = None

    def get_train_example_dict(self):
        return self.corpus.train_example_dict

    def get_test_example_dict(self):
        return self.corpus.test_example_dict

    def get_max_sent_len(self):
        return self.corpus.get_max_sent_len()

    def get_max_dep_type(self):
        return self.corpus.get_dep_type_count()

    def get_sentence_by_id(self, s_id):
        sentence = self.corpus.get_sentence_by_id[str(s_id)]
        return sentence.original

    def get_sentence_pair_by_id(self, sentence_ids):
        return self.get_sentence_by_id(sentence_ids[0]), self.get_sentence_by_id(sentence_ids[1])

    # def get_sentence_pair_by_e_id(self, e_id, example_dict):
    #     return self.get_sentence_by_id(sentence_ids[0]), self.get_sentence_by_id(sentence_ids[1])

    def list_of_train_loader_tuple(self, k_fold, batch_size):
        if k_fold == 1:
            return ValueError

        if k_fold < 0:
            train_loader = self.create_loader(MyDateSet(self.train_example_manager.examples), batch_size=batch_size,
                                              example_dict=self.corpus.train_example_dict)
            valid_loader = self.create_loader(MyDateSet(self.dev_example_manager.examples), batch_size=batch_size,
                                              example_dict=self.corpus.train_example_dict)
            loader_tuple = (train_loader, valid_loader)

            result = loader_tuple,

        else:
            if DataManager.example_groups is None:
                DataManager.example_groups = self.train_example_manager.divide_into_groups_keep_rate(k_group=k_fold)
            example_groups = DataManager.example_groups
            self.check_example_groups(example_groups)
            result = []
            train_group = []
            valid_group = []
            for i in range(k_fold):
                train_group.clear()
                valid_group.clear()
                for j in range(k_fold):
                    if j == i:
                        valid_group = example_groups[j].copy()
                    else:
                        train_group.extend(example_groups[j].copy())
                if len(train_group) == 0 or len(valid_group) == 0:
                    raise ValueError
                train_loader = self.create_loader(MyDateSet(train_group.copy()), batch_size=batch_size, example_dict=self.corpus.train_example_dict)
                valid_loader = self.create_loader(MyDateSet(valid_group.copy()), batch_size=batch_size, example_dict=self.corpus.train_example_dict)
                loader_tuple = (train_loader, valid_loader)
                # print('train_loader:{}  valid_loader:{}'.format(len(train_loader), len(valid_loader)))
                # self.check_data_loader_tuple(loader_tuple)
                result.append(loader_tuple)
            # self.check_data_loader_tuple_list(result)
        return result

    def create_loader(self, data_set, drop_last=False, batch_size=2, shuffle=True, example_dict=None):
        data_loader = MyDataLoader(data_set, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
        data_loader.example_dict = example_dict
        return data_loader

    def test_loader(self, batch_size):
        return self.create_loader(MyDateSet(self.test_example_manager.examples), batch_size=batch_size, example_dict=self.corpus.test_example_dict)

    def train_loader(self, batch_size):
        return self.create_loader(MyDateSet(self.train_example_manager.examples), batch_size=batch_size, example_dict=self.corpus.train_example_dict)

    def get_loader_dict(self, k_fold=5, batch_size=1, force=False):
        if (self.loader_dict is None) or force:
            global_loader_dict = {}
            train_loader_tuple_list = self.list_of_train_loader_tuple(k_fold=k_fold, batch_size=batch_size)
            global_loader_dict['train_loader_tuple_list'] = train_loader_tuple_list

            test_loader = self.test_loader(batch_size=batch_size)
            global_loader_dict['test_loader'] = test_loader
            self.loader_dict = global_loader_dict
        return self.loader_dict

    def calculate_distribution_of_two_class(self, labels):
        negative_count = 0
        positive_count = 0
        for label in labels:
            if label == 1:
                positive_count += 1
            elif label == 0:
                negative_count += 1
            else:
                raise ValueError
        return positive_count, negative_count

    def check_example_groups(self, example_dict_groups):
        example_id_set = []
        example_id_dict = {}
        example_count = 0
        example_rates = []
        for group in example_dict_groups:
            p, n = self.calculate_distribution_of_two_class([int(example.label) for example in group])
            example_rate = round(p/n, 2)
            example_rates.append((p, n, example_rate))
            for example in group:
                example_id_set.append(str(example.id))
                if str(example.id) in example_id_dict:
                    raise ValueError
                example_id_dict[str(example.id)] = example_count
                example_count += 1
        rates = []
        for rate in example_rates:
            rates.append(rate[-1])
        print(rates)
        rates = np.array(rates)
        if rates.var() > 0.1:
            raise ValueError
        if len(set(example_id_set)) != example_count:
            raise ValueError
        if len(example_id_dict.keys()) != example_count:
            raise ValueError
        if len(set(example_id_set)) != len(example_id_dict.keys()):
            raise ValueError

    def check_data_loader_tuple(self, loaders):
        train_loader, valid_loader = loaders
        train_examples = train_loader.dataset.examples
        valid_examples = valid_loader.dataset.examples
        example_id_dict = {}
        example_count = 0
        for example in train_examples:
            example_id_dict[str(example.id)] = example_count
            example_count += 1

        for example in valid_examples:
            example_id_dict[str(example.id)] = example_count
            example_count += 1

        if len(example_id_dict.keys()) != example_count:
            raise ValueError

        if len(self.train_example_manager.examples) != len(example_id_dict.keys()):
            raise ValueError

        if len(train_examples)+len(valid_examples) != len(self.train_example_manager.examples):
            raise ValueError

        for example in self.train_example_manager.examples:
            if str(example.id) not in example_id_dict:
                raise ValueError

    def check_data_loader_tuple_list(self, data_loader_tuple_list):
        def check_valid_two_tuple(loader_tuple1, loader_tuple2):
            train_loader1, valid_loader1 = loader_tuple1
            train_loader2, valid_loader2 = loader_tuple2

            train_example_id_dict1 = {}
            train_example_id_dict2 = {}

            valid_example_id_dict1 = {}
            valid_example_id_dict2 = {}

            for example in train_loader1.dataset:
                train_example_id_dict1[str(example['example_id'].item())] = 11
            for example in valid_loader1.dataset:
                valid_example_id_dict1[str(example['example_id'].item())] = 12

            for example in train_loader2.dataset:
                train_example_id_dict2[str(example['example_id'].item())] = 21
            for example in valid_loader2.dataset:
                valid_example_id_dict2[str(example['example_id'].item())] = 22
            # count = 0
            for key in valid_example_id_dict2.keys():
                if key not in train_example_id_dict1:
                    raise ValueError

            for key in valid_example_id_dict1:
                if key not in train_example_id_dict2:
                    raise ValueError

            for key in valid_example_id_dict1.keys():
                if key in valid_example_id_dict2:
                    raise ValueError
            # print(count)

        list_count = len(data_loader_tuple_list)
        for i in range(list_count-1):
            for j in range(i+1, list_count):
                check_valid_two_tuple(data_loader_tuple_list[i], data_loader_tuple_list[j])


def align_sentence_tokens(sentence_token, max_sentence_len, unk_token, direction='left'):
    result = list(sentence_token.copy())
    for i in range(len(result), max_sentence_len):
        if direction == 'left':
            result.append(unk_token)
        elif direction == 'right':
            result.insert(0, unk_token)
        else:
            raise ValueError("Unknow direction parameter")
    return tuple(result)


def align_mult_sentence_tokens(mult_sentence_tokens, max_sentence_len, unk_token, direction='left'):
    result = []
    for sentence_tokes in mult_sentence_tokens:
        result.append(align_sentence_tokens(sentence_tokes, max_sentence_len, unk_token, direction=direction))
    for t in result:
        if len(t) != len(result[0]):
            raise ValueError("align defeat!")
    return result


def padding_tensor(tensor_, max_sentence_len, align_dir='left', dim=0):
    padding_len = max_sentence_len - len(tensor_)
    if padding_len < 0:
        raise ValueError("padding max len smaller")
        # indexes = [i for i in range(max_sentence_len)]
        # indexes = torch.LongTensor(indexes).to(tensor_.device)
        # tensor_ = tensor_.index_select(dim=dim, index=indexes)
        # return tensor_

    shape = torch.tensor(tensor_.size(), dtype=torch.int).tolist()
    shape[dim] = padding_len
    pad_tensor = torch.zeros(shape, dtype=tensor_.dtype, device=tensor_.device)
    if align_dir == 'left':
        tensor_ = torch.cat([tensor_, pad_tensor], dim=dim)
    elif align_dir == 'right':
        tensor_ = torch.cat([pad_tensor, tensor_], dim=dim)
    else:
        raise ValueError("Unknow direction parameter")
    return tensor_

