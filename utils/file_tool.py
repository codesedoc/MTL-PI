import os
import pickle
import csv


def get_absolute_path(file):
    return os.path.dirname(file)


def dirname(file):
    return get_absolute_path(file)


def load_data(file_name, mode, errors=None):
    with open(file_name, mode, errors=errors) as f:
        rows = f.readlines()
    return rows


def is_file(file):
    return os.path.isfile(file)


def save_data(data, file_name, model):
    with open(file_name, model) as f:
        f.write(data)


def save_list_data(data, file_name, model, need_new_line=True):
    with open(file_name, model) as f:
        for d in data:
            f.write(d)
            if need_new_line:
                f.write('\n')


def save_data_pickle(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def load_data_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def connect_path(pre_path, *part_paths):
    parts = part_paths
    for p in parts:
        pre_path = os.path.join(pre_path, p)
    return pre_path


def exists(path):
    return os.path.exists(path)


def makedir(path):
    if not os.path.exists(path):
        return os.makedirs(path)


def check_dir(path):
    return os.path.exists(path)


def list_dir(path):
    return os.listdir(path)


def check_file(filename):
    return os.path.exists(filename)


def read_csv(input_file, delimiter=',', quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))


def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


def write_lines_to_tsv(data_list_lines, input_file, quotechar=None):
    with open(input_file, 'w') as f:
        tsv_w = csv.writer(f, delimiter='\t', quotechar=quotechar)
        for line in data_list_lines:
            tsv_w.writerow(line)


class FileOperator:
    def __init__(self, file_name):
        self.file_name = file_name

    def read(self, operation_func, model):
        result = load_data(self.file_name,model)
        return operation_func(result)

    def write(self, all_data, operation_func, model):
        all_data = operation_func(all_data)
        save_data(all_data,self.file_name, model)



class PathManager:
    model_running_data_log_file = 'result/log.txt'
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # data_path = os.path.join(base_dir, 'data')

    # glove_path = os.path.join(data_path, 'glove')
    # glove_embedding_file_path = os.path.join(glove_path, 'glove.840B.300d.txt')
    # glove_embedding_test_file_path = os.path.join(glove_path, 'glove.840B.300d_test.txt')
    # glove_dictionary_file_path = os.path.join(glove_path, 'glove_dictionary.pkl')
    # glove_dictionary_test_file_path = os.path.join(glove_path, 'glove_dictionary_test.pkl')

    # msrpc_path = os.path.join(data_path, 'msrpc')
    # msrpc_train_data_set_path = os.path.join(msrpc_path, 'train.txt')
    # msrpc_test_data_set_path = os.path.join(msrpc_path, 'test.txt')
    # msrpc_train_data_manager_path = os.path.join(msrpc_path, 'train_data_manager.pkl')
    # msrpc_test_data_manager_path = os.path.join(msrpc_path, 'test_data_manager.pkl')

    result_path = os.path.join(base_dir, 'result')
    # log_path = os.path.join(result_path,'log')
    # error_glove_embedding_data_file = os.path.join(log_path, 'error_glove_embedding_data.txt')
    # word_no_in_dictionary_file = os.path.join(log_path, 'word_no_in_dictionary.txt')
    # model_log_path =  os.path.join(log_path, "model_log")
    # model_running_data_log_file = os.path.join(model_log_path, "model_running_data")

    model_path = os.path.join(result_path,'model')
    entire_model_file =  os.path.join(model_path, 'entire_model.pkl')

    visualization_path = os.path.join(result_path, 'visualization')
    tensorboard_runs_path = os.path.join(visualization_path, 'tensorboard_runs')

    @staticmethod
    def change_filename_by_append(file_path, append_str):
        (dir_path, file_name_ext) = os.path.split(file_path)
        (filename, extension) = os.path.splitext(file_name_ext)
        new_path = dir_path+r'/'+filename + '_' + append_str +extension
        return new_path

    @staticmethod
    def append_filename_to_dir_path(dir_path, filename, extent='txt'):
        return os.path.join(dir_path, filename+'.'+extent)