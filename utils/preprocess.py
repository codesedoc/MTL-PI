import utils.file_tool as file_tool

class StandardPOSDepPair:
    def __init__(self,  **kwargs):
        self.pos_first = kwargs['pos_first']
        self.pos_second = kwargs['pos_second']
        self.dep_names = kwargs['dep_names']


def create_pos_dep_pair_dict(org_file, output_file=None):
    ### create dictionary ###
    pos_dep_pairs_dict = {}
    rows = file_tool.load_data(org_file, 'r')
    for i, row in enumerate(rows):
        items = row.split()
        if len(items) != 4:
            raise ValueError
        key = "%s_%s" % (items[0], items[1])

        if key not in pos_dep_pairs_dict:
            standardPDP = StandardPOSDepPair(pos_first=items[0], pos_second=items[1], dep_names=[items[2]])
            pos_dep_pairs_dict[key] = standardPDP
        else:
            standardPDP = pos_dep_pairs_dict[key]
            standardPDP.dep_names.append(items[2])

    #### check ####
    row_count = 0
    for standardPDP in pos_dep_pairs_dict.values():
        row_count += len(standardPDP.dep_names)
    if row_count != len(rows):
        raise ValueError

    ### save dictionary ###
    if output_file is not None:
        save_data = []
        for standardPDP in pos_dep_pairs_dict.values():
            save_data.append("\t".join([standardPDP.pos_first, standardPDP.pos_second, str(standardPDP.dep_names)]))
        file_tool.save_list_data(save_data, output_file, 'w')

    return pos_dep_pairs_dict


def test():
    create_pos_dep_pair_dict('data_preprocess/standard_pos_dep_pairs.txt', 'data_preprocess/standard_pos_dep_list.txt')