import math
import numpy as np
import torch
import random
import utils.file_tool as file_tool


def singleton(cls):
    _instance = None
    def _singleton(*args, **kargs):
        nonlocal _instance
        if _instance == None:
            _instance = cls(*args, **kargs)
        return _instance
    # print('id{}'.format(id(_singleton)))
    return _singleton

def compare_two_dict_keys(dict1, dict2):
    for key in dict1.keys():
        if key not in dict2:
            return False

    for key in dict2.keys():
        if key not in dict1:
            return False
    return True


global_position_encodings = None


def get_position_encodings(length, dimension):
    position_encodings = []
    for t in range(length):
        position_encoding = []
        for d in range(dimension):
            if d % 2 == 0:
                d = int(d / 2)
                wd = 1/(math.pow(10000, 2*d/dimension))
                position_encoding.append(math.sin(wd * t))
                position_encoding.append(math.cos(wd * t))
        position_encodings.append(position_encoding)
    position_encodings = np.array(position_encodings)
    return position_encodings


def get_global_position_encodings(length=100, dimension=300):
    global global_position_encodings
    if global_position_encodings is None:
        global_position_encodings = get_position_encodings(length, dimension)
    return global_position_encodings


# def get_sinusoid_encoding_table(n_position, d_hid):
#     ''' Sinusoid position encoding table '''
#
#     def cal_angle(position, hid_idx):
#         return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)
#
#     def get_posi_angle_vec(position):
#         return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
#
#     sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
#
#     sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i  偶数正弦
#     sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1  奇数余弦
#     return sinusoid_table


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def word_piece_flag_list(revised_tokens, split_signal):
    flag_list = np.zeros(len(revised_tokens), dtype=np.int)
    range_list = []
    start = -1
    end = -1
    word_piece_label = False
    for i, token in enumerate(revised_tokens):
        token = token.strip()
        if len(token) == 0:
            raise ValueError
        if token.startswith(split_signal):
            if i <= 0:
                raise ValueError
            if not word_piece_label:
                start = i-1
            word_piece_label = True
        else:
            end = i
            if word_piece_label:
                if end < start+1:
                    raise ValueError
                range_list.append((start, end))

            word_piece_label = False
    if word_piece_label:
        range_list.append((start, len(revised_tokens)))

    for range_ in range_list:
        for i in range(range_[0], range_[1]):
            flag_list[i] = 1
    return flag_list


def calculate_the_max_len_of_tokens_split_by_bert(corpus_obj, tokenizer):
    sentence_dict = corpus_obj.sentence_dict_from_examples()
    max_len = 0
    max_len_id = 0
    # sentence_dict = {'233158': sentence_dict["233158"], '233159': sentence_dict["233159"]}
    count = 0
    input_is_list = []
    for sent_id, sentence in sentence_dict.items():
        inputs_ls_cased = tokenizer.encode_plus(sentence.sentence_with_root_head())
        input_ids = inputs_ls_cased["input_ids"]
        input_is_list.append(input_ids)
        revised_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        if len(revised_tokens) >= 126:
            count += 1
        if len(revised_tokens) > max_len:
            max_len = len(revised_tokens)
            max_len_id = sent_id
        # print(len(revised_tokens))
    print("the_max_len_of_tokens_split_by_bert is {}, and id is {}".format(max_len, max_len_id))
    print("the_count_of_the_len_larger_than_126 is {}".format(count))
    # s1 = sentence_dict["233158"]
    # s2 = sentence_dict["233159"]
    # inputs_ls_cased = tokenizer.encode_plus(s1.sentence_with_root_head(), s2.sentence_with_root_head(),
    #                                         add_special_tokens=True,
    #                                         max_length=256,)
    # input_ids, token_type_ids = inputs_ls_cased["input_ids"], inputs_ls_cased["token_type_ids"]

    # piece_flag_list = word_piece_flag_list(tokenizer.convert_ids_to_tokens(input_ids), '##')

    pass


def covert_transformer_tokens_to_words(corpus_obj, tokenizer,  result_file, split_signal):
    sentence_dict = corpus_obj.sentence_dict
    words_dict = {}
    special_words_dict = {}
    for sent_id, sentence in sentence_dict.items():
        inputs_ls_cased = tokenizer.encode_plus(sentence.original_sentence())
        input_ids = inputs_ls_cased["input_ids"]
        revised_tokens = tokenizer.convert_ids_to_tokens(input_ids)

        piece_flag_list = word_piece_flag_list(revised_tokens[1:-1].copy(), split_signal)

        words_temp = revised_tokens[1:-1].copy()
        for i, token in enumerate(words_temp):
            token = token.strip()
            if len(token) == 0:
                raise ValueError
            if token.startswith(split_signal):
                words_temp[i] = token[2:]

        if len(piece_flag_list) != len(words_temp):
            raise ValueError

        words = []
        word_piece_label = False
        word_piece_list = []
        for i, flag in enumerate(piece_flag_list):
            if flag == 1:
                word_piece_list.append(words_temp[i])
                word_piece_label = True
            else:
                if word_piece_label:
                    if len(word_piece_list) == 0:
                        raise ValueError
                    words.append(''.join(word_piece_list))
                    word_piece_list.clear()

                words.append(words_temp[i])
                word_piece_label = False
        if word_piece_label and len(word_piece_list) > 0:
            words.append(''.join(word_piece_list))
            special_words_dict[sent_id] = words

        words_dict[sent_id] = words

    save_data = []
    for sent_id, words in words_dict.items():
        save_data.append(sent_id + '\t' + ' '.join(words))
    file_tool.save_list_data(save_data, result_file, 'w')

    save_data = []
    for sent_id, words in special_words_dict.items():
        save_data.append(sent_id + '\t' + ' '.join(words))
    file_tool.save_list_data(save_data, file_tool.PathManager.change_filename_by_append(result_file, 'special'), 'w')

    pass

# def covert_transformer_tokens_to_words(corpus_obj, tokenizer,  result_file, split_signal):
#     sentence_dict = corpus_obj.sentence_dict
#     words_dict = {}
#     for sent_id, sentence in sentence_dict.items():
#         inputs_ls_cased = tokenizer.encode_plus(sentence.original_sentence())
#         input_ids = inputs_ls_cased["input_ids"]
#         revised_tokens = tokenizer.convert_ids_to_tokens(input_ids)
#         words = []
#         word_piece_label = False
#         word_piece_list = []
#         for token in revised_tokens[1:-1]:
#             token = token.strip()
#             if len(token) == 0:
#                 raise ValueError
#             if token.startswith(split_signal):
#                 word_piece_list.append(token[2:])
#                 word_piece_label = True
#             else:
#                 if word_piece_label:
#                     if len(word_piece_list) == 0:
#                         raise ValueError
#                     if len(words) == 0:
#                         raise ValueError
#                     word_temp = words.pop()
#                     words.append(word_temp + ''.join(word_piece_list))
#                     word_piece_list.clear()
#                 else:
#                     words.append(token)
#                 word_piece_label = False
#
#         words_dict[sent_id] = words
#     save_data = []
#     for sent_id, words in words_dict.items():
#         save_data.append(sent_id + '\t' + ' '.join(words))
#     file_tool.save_list_data(save_data, result_file, 'w')
#     pass


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test():
    result1 = get_global_position_encodings(10, 6)
    # result2 = get_sinusoid_encoding_table(10, 6)
    pass


if __name__ == '__main__':
    test()

