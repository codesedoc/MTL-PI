from data.corpus import *

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/fukumoto/Documents/study/code/python/MTL_PI')

    mrpc = MRPCorpus()

    mrpc.load_all_data()
    print(mrpc.data_distribute)