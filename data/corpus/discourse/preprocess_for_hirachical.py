import utils.file_tool as file_tool
import random
import math

def add_index_to_segment_in_source_files():
    # file_path = 'raw/annotator1'
    file_path = 'raw/annotator2'
    file_number_range = range(1, 136)
    for i in file_number_range:
        text_name = file_tool.connect_path(file_path, str(i))
        indexed_text = file_tool.connect_path(file_path, str(i) + "-indexed")
        content = file_tool.load_data(text_name, 'r', errors='ignore')
        discourse_segments = []
        index = 0
        # extra segments
        for row in content:
            if len(row.strip()) != 0:
                row = "%d\t%s" % (index, row)
                index+=1
            discourse_segments.append(row)
        file_tool.save_list_data(discourse_segments, indexed_text, 'w', need_new_line=False)

def create_examples_sentences(file_path):
    file_number_range = range(1, 136)
    examples = []
    elab_examples = []
    for i in file_number_range:
        examples_from_one_file = []
        text_name = file_tool.connect_path(file_path, str(i))
        annotation_name = file_tool.connect_path(file_path, "%d-%s" % (i, 'annotation'))

        content = file_tool.load_data(text_name, 'r',  errors='ignore')
        discourse_segments = []

        # extra segments
        for row in content:
            row = row.strip()
            if len(row) == 0:
                continue
            discourse_segments.append(row)

        content = file_tool.load_data(annotation_name, 'r',  errors='ignore')

        # extra annotations
        annotations = []
        for row in content:
            row = row.strip()
            if len(row) == 0:
                continue
            annotations.append(row)

        # create example
        for anno in annotations:
            items = anno.split(' ')
            if len(items) != 5:
                raise ValueError
            label = items[-1]
            e_label = label.split('-')[0]
            # if label.startswith('elab') and label.split('-')[0] != 'elab':
            #     raise ValueError

            if (int(items[0]) > int(items[1])) or (int(items[2]) > int(items[3])):
                continue

            if (int(items[1]) >= len(discourse_segments)) or (int(items[3]) >= len(discourse_segments)):
                continue

            satellite_index = (int(items[0]), int(items[1]))
            nucleus_index = (int(items[2]), int(items[3]))

            # if label.startswith('elab'):
            #     e_label = label
            # else:
            #     e_label = 0
            satellite = ' '.join(discourse_segments[satellite_index[0]: satellite_index[1]+1])
            nucleus = ' '.join(discourse_segments[nucleus_index[0]: nucleus_index[1] + 1])

            if satellite == '' or nucleus == '':
                raise ValueError

            e = {
                'satellite': satellite,
                'nucleus': nucleus,
                'label': e_label,
                'relation': label,
                'source': 'annotator1-' + str(i) + '-(' + anno + ')',
                'file_number': i,
                'anno_info': items
            }
            examples_from_one_file.append(e)
            examples.append(e)
            if e['label'] == 1:
                elab_examples.append(e)

    # print("the total examples: {}, the elaboration examples: {}".format(len(examples), len(elab_examples)))

    segment_pair2example = {}
    for e in examples:
        # pair_temp =
        segment_pair_info = "%d\t%s" % (e['file_number'], '-'.join(e['anno_info'][:-1]))
        if segment_pair_info in segment_pair2example:
            pass
        segment_pair2example[segment_pair_info] = e

    print("The number of repeat pair but in same segment file: {} at path:{}".format(
        (len(examples) - len(segment_pair2example)), file_path
    ))
    examples = []
    for e in segment_pair2example.values():
        examples.append(e)

    if len(examples) != len(segment_pair2example):
        raise ValueError

    return examples, segment_pair2example


def create_category2examples_and_sentences():
    annotator1_path = 'raw/annotator1'
    annotator1_examples, annotator1_segment_pair2example = create_examples_sentences(annotator1_path)

    annotator2_path = 'raw/annotator2'
    annotator2_examples, annotator2_segment_pair2example = create_examples_sentences(annotator2_path)

    category2examples = {}
    elab_examples = []
    examples = []

    for pair, example_from_anno1 in annotator1_segment_pair2example.items():
        if pair in annotator2_segment_pair2example:
            example_from_anno2 = annotator2_segment_pair2example[pair]
            if example_from_anno1['label'] == example_from_anno2['label']:
                label = example_from_anno1['label']
                example = example_from_anno1
                if label not in category2examples:
                    category2examples[label] = []
                category2examples[label].append(example)
                if label == 'elab':
                    elab_examples.append(example)
                examples.append(example)

    num_of_all_examples = 0
    for es in category2examples.values():
        num_of_all_examples += len(es)

    if num_of_all_examples != len(examples):
        raise ValueError

    print("In this corpus, the total examples: {}, the elaboration examples: {}, the non-elaboration examples: {}".format(
        num_of_all_examples, len(elab_examples), num_of_all_examples - len(elab_examples)
    ))

    # create sentences and update the ids of sentences in examples

    sentences = set()
    sentences2id = {}
    for e in examples:
        satellite = e['satellite']
        nucleus = e['nucleus']

        sentences.add(satellite)
        sentences.add(nucleus)

    sentences = list(sentences)

    for id_, s in enumerate(sentences):
        sentences2id[s] = id_

    e_id_count = 0
    for e in examples:
        satellite = e['satellite']
        e['satellite_id'] = sentences2id[satellite]
        nucleus = e['nucleus']
        e['nucleus_id'] = sentences2id[nucleus]
        e['id'] = e_id_count
        e_id_count += 1

    #save sentence and examples
    save_data = ['\t'.join(('id', 'text'))]
    for sent, sent_id in sentences2id.items():
        save_data.append("{}\t{}".format(str(sent_id), sent))

    file_tool.save_list_data(save_data, 'raw/sentences.txt', 'w')

    save_examples(examples, 'raw/examples.txt')
    # return examples, sentences

    save_data = ['\t'.join(('type', 'count'))]
    category2examples_itmes_sorted = sorted(category2examples.copy().items(), key=lambda item: len(item[1]), reverse=True)
    non_elab_count = 0
    for type_name, es in category2examples_itmes_sorted:
        save_data.append(f'{type_name}\t{len(es)}')
        if type_name != "elab":
            non_elab_count += len(es)
    print("non_elab_count: %d" % non_elab_count )
    file_tool.save_list_data(save_data, 'raw/category_count.txt', 'w')

    return category2examples, sentences


def save_examples(examples, file_name):
    save_data = [('id', 'label', 'satellite_id', 'satellite', 'nucleus_id', 'nucleus', 'source')]
    for e in examples:
        save_data.append((
            str(e['id']),
            str(e['label']),
            str(e['satellite_id']),
            e['satellite'],
            str(e['nucleus_id']),
            e['nucleus'],
            # e['relation'],
            e['source']
        ))

    file_tool.write_lines_to_tsv(save_data, file_name)


def _sample_from_list(org_list, sample_rate):

    org_num = len(org_list)
    sample_indexes = set()
    while (len(sample_indexes) <= math.ceil(len(org_list) * sample_rate)):
        try:
            sample_indexes.add(random.randint(0, len(org_list)-1))
        except Exception:
            raise
    samples = []
    example_list = org_list.copy()
    for index in sample_indexes:
        samples.append(example_list[index])
        org_list.remove(example_list[index])

    org_examples_id_set = set()
    samples_id_set = set()

    for e in org_list:
        org_examples_id_set.add(e['id'])
    for e in samples:
        samples_id_set.add(e['id'])

    if len(org_examples_id_set) + len(samples_id_set) != org_num:
        raise ValueError
    return samples


def divide_examples(examples):
    non_elaboration_e = []
    elaboration_e = []
    label = examples[0]['label']
    for e in examples:
        if label != e['label']:
            raise ValueError

    dev_rate = 0.1
    test_rate = 0.25
    train_set = examples.copy()
    test_set = _sample_from_list(train_set, test_rate)
    dev_set = _sample_from_list(train_set, dev_rate)
    if len(train_set) + len(test_set) + len(dev_set) != len(examples):
        raise ValueError
    return {'train': train_set, 'dev': dev_set, 'test': test_set}


def divide_all_examples(category2examples):
    result = {}
    for type_name, es in category2examples.items():
        result[type_name] = divide_examples(es)
    return result


def count_examples(examples):
    result = {}
    e_id_set = set()
    for e in examples:
        label = e['label']
        if label not in result:
            result[label] = 0
        result[label] += 1
        e_id_set.add(int(e['id']))
    if len(examples) != len(e_id_set):
        raise ValueError
    rate = {}
    for type_name, count in result.items():
        rate[type_name] = round(count/len(e_id_set), 3)
    print(f"The rate of categories: {rate}")
    return result, e_id_set


def merger_examples_divided_list(examples_divided_list):
    set_name_set = ('train', 'dev', 'test')
    result = {'train': [], 'dev': [], 'test': []}

    for esed in examples_divided_list:
        for set_name, es in esed.items():
            if set_name not in set_name_set:
                raise ValueError
            result[set_name].extend(es)

    all_e_id_set = set()
    count = 0
    for set_name, es in result.items():
        print()
        print(f'This data set have {len(es)} examples belong to {set_name}')
        category_cout, e_id_set = count_examples(es)
        print(f'{set_name} data: the category distribution-{category_cout}')
        all_e_id_set.update(e_id_set)
        count += len(e_id_set)
    if len(all_e_id_set) != count:
        raise ValueError
    print(f'The number of all example belong to this kind data sets is:{count}')
    return result, all_e_id_set


def run():
    level1 = ('temp', 'attr', 'ce')
    level2 = ('par', 'contr', 'elab')

    category2examples, _ = create_category2examples_and_sentences()
    for name in category2examples.copy():
        if (name not in level1) and (name not in level2):
            category2examples.pop(name)

    category2examples_divided = divide_all_examples(category2examples)

    if len(category2examples_divided) != 6:
        raise ValueError

    total_id2example = {}

    import copy
    elab_e_id_set = set()
    not_elab_e_id_set = set()
    for key,esed in category2examples_divided.items():
        if (key not in level1) and (key not in level2):
            raise ValueError
        for set_name in esed:
            for e in esed[set_name]:
                if e['id'] in total_id2example:
                    raise ValueError
                total_id2example[e['id']] = e
                if e['label'] == 'elab':
                    elab_e_id_set.add(int(e['id']))
                else:
                    not_elab_e_id_set.add(int(e['id']))

    original_total_id2example = copy.deepcopy(total_id2example)
    print(f'The total examples utilized now: {len(total_id2example)}')

    # save all example to resemblance data sets
    print('\n' + '*'*80)
    print('Create resemblance data sets')
    level2_esed_list = [category2examples_divided[type_name] for type_name in level2]

    resemblance_sets, resemblance_e_id_set = merger_examples_divided_list(level2_esed_list)

    # check set_type
    set_name2e_id_set = {}
    for esed in level2_esed_list:
        for set_name, es in esed.items():
            set_temp = set()
            for e in es:
                set_temp.add(e['id'])
            if set_name not in set_name2e_id_set:
                set_name2e_id_set[set_name] = set()
            set_name2e_id_set[set_name].update(set_temp)

    for set_name, id_set in set_name2e_id_set.items():
        for id_ in id_set:
            for set_name_, id_set_ in set_name2e_id_set.items():
                if set_name == set_name_:
                    continue
                if id_ in id_set_:
                    raise ValueError

    for set_name, es in resemblance_sets.items():
        set_temp = set_name2e_id_set[set_name]
        for e in es:
            if e['id'] not in set_temp:
                raise ValueError
        # really save
        save_examples(es, f'resemblance/raw/{set_name}.txt')
        for e in es:
            e['label'] = 'resemblance'
    print('*'*80)

    # save all example to root data sets
    print('\n' + '*'*80)
    print('Create root data sets')
    level1_esed_list = [category2examples_divided[type_name] for type_name in level1]
    level1_esed_list.append(resemblance_sets)

    root_sets, root_e_id_set = merger_examples_divided_list(level1_esed_list)

    # check set_type
    set_name2e_id_set = {}
    for esed in level1_esed_list:
        for set_name, es in esed.items():
            set_temp = set()
            for e in es:
                set_temp.add(e['id'])
            if set_name not in set_name2e_id_set:
                set_name2e_id_set[set_name] = set()
            set_name2e_id_set[set_name].update(set_temp)

    for set_name, id_set in set_name2e_id_set.items():
        for id_ in id_set:
            for set_name_, id_set_ in set_name2e_id_set.items():
                if set_name == set_name_:
                    continue
                if id_ in id_set_:
                    raise ValueError

    for set_name, es in root_sets.items():
        set_temp = set_name2e_id_set[set_name]
        for e in es:
            if e['id'] not in set_temp:
                raise ValueError
        # really save
        save_examples(es, f'coherence/raw/{set_name}.txt')

    if len(root_e_id_set) != len(total_id2example):
        raise ValueError
    print('*'*80)

    # save all example to elab or non-elab
    print('\n' + '*'*80)
    print('Create elaboration data sets')
    total_num_elab = 0
    total_num_no_elab = 0
    for set_name, es in root_sets.items():
        num_elab_es = 0
        for e in es:
            if e['id'] in elab_e_id_set:
                e['label'] = 1
                num_elab_es += 1
            elif e['id'] in not_elab_e_id_set:
                e['label'] = 0
            else:
                raise ValueError
        num_no_elab = len(es)-num_elab_es
        total_num_elab += num_elab_es
        total_num_no_elab += num_no_elab
        print(f'{set_name} data, the number of examples: {len(es)},   "elab": {num_elab_es},  "no-elab":  {num_no_elab}, '
              f'the rate is:{round(num_elab_es/len(es),2)} ,  {round(num_no_elab/len(es),2)}')
        save_examples(es, f'elaboration/raw/{set_name}.txt')
    if (total_num_elab != len(elab_e_id_set)) or (total_num_no_elab != len(not_elab_e_id_set)):
        raise ValueError
    print('*'*80)

if __name__ ==  "__main__":
    from utils import general_tool
    general_tool.setup_seed(1)
    run()
# #
#     from process_source_data import create_examples_sentences as old_create_examples
#     annotator1_path = 'raw/annotator1'
#     annotator_examples, annotator_segment_pair2example = create_examples_sentences(annotator1_path)
#     annotator_examples_old, annotator_segment_pair2example_old = old_create_examples(annotator1_path)
#
#     for key,e in annotator_segment_pair2example.items():
#         if key not in annotator_segment_pair2example_old:
#             raise ValueError
#         if annotator_segment_pair2example_old[key] != e:
#             raise ValueError
