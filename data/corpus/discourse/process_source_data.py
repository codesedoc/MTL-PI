import utils.file_tool as file_tool
import random
import math

def add_index_to_segment_in_source_files():
    # file_path = 'corpus/discourse/annotator1'
    file_path = 'corpus/discourse/annotator2'
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

            if (int(items[0]) > int(items[1])) or (int(items[2]) > int(items[3])):
                continue

            if (int(items[1]) >= len(discourse_segments)) or (int(items[3]) >= len(discourse_segments)):
                continue

            satellite_index = (int(items[0]), int(items[1]))
            nucleus_index = (int(items[2]), int(items[3]))

            if label.startswith('elab'):
                e_label = 1
            else:
                e_label = 0
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


def create_examples_and_sentences_files():
    annotator1_path = 'raw/annotator1'
    annotator1_examples, annotator1_segment_pair2example = create_examples_sentences(annotator1_path)

    annotator2_path = 'raw/annotator2'
    annotator2_examples, annotator2_segment_pair2example = create_examples_sentences(annotator2_path)

    examples = []
    elab_examples = []
    for pair, example_from_anno1 in annotator1_segment_pair2example.items():
        if pair in annotator2_segment_pair2example:
            example_from_anno2 = annotator2_segment_pair2example[pair]
            if example_from_anno1['label'] == example_from_anno2['label']:
                examples.append(example_from_anno1)
                if example_from_anno1['label'] == 1:
                    elab_examples.append(example_from_anno1)
                elif example_from_anno1['label'] == 0:
                    pass
                else:
                    raise ValueError

    print("the total examples: {}, the elaboration examples: {}, the non-elaboration examples: {}".format(
        len(examples), len(elab_examples), len(examples) - len(elab_examples)
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

    # save sentence and examples

    save_data = ['\t'.join(('id', 'text'))]
    for sent, sent_id in sentences2id.items():
        save_data.append("{}\t{}".format(str(sent_id), sent))

    file_tool.save_list_data(save_data, 'raw/data/sentences.txt', 'w')

    # save_data = ['\t'.join(('id', 'label', 'satellite_id', 'satellite', 'nucleus_id', 'nucleus', 'source'))]
    # for e in examples:
    #     save_data.append("\t".join([
    #         str(e['id']),
    #         str(e['label']),
    #         str(e['satellite_id']),
    #         e['satellite'],
    #         str(e['nucleus_id']),
    #         e['nucleus'],
    #         # e['relation'],
    #         e['source']
    #     ]))
    #
    # file_tool.save_list_data(save_data, 'corpus/discourse/data/examples.txt', 'w')

    save_examples(examples, 'raw/data/examples.txt')
    return examples, sentences

def save_examples(examples, file_name):
    save_data = ['\t'.join(('id', 'label', 'satellite_id', 'satellite', 'nucleus_id', 'nucleus', 'source'))]
    for e in examples:
        save_data.append("\t".join([
            str(e['id']),
            str(e['label']),
            str(e['satellite_id']),
            e['satellite'],
            str(e['nucleus_id']),
            e['nucleus'],
            # e['relation'],
            e['source']
        ]))

    file_tool.save_list_data(save_data, file_name, 'w')

def _sample_from_list(org_list, sample_rate):
    org_num = len(org_list)
    sample_indexes = set()
    while (len(sample_indexes) <= math.ceil(len(org_list) * sample_rate)):
        sample_indexes.add(random.randint(0, len(org_list)-1))
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


def divide_examples():
    examples, _ = create_examples_and_sentences_files()
    non_elaboration_e = []
    elaboration_e = []
    for e in examples:
        if e['label'] == 1:
            elaboration_e.append(e)
        elif e['label'] == 0:
            non_elaboration_e.append(e)
        else:
            raise ValueError
    dev_rate = 0.1
    test_rate = 0.25
    train_set = examples.copy()
    test_set = _sample_from_list(train_set, test_rate)
    dev_set = _sample_from_list(train_set, dev_rate)
    save_examples(train_set, 'raw/data/train_set.txt')
    save_examples(test_set, 'raw/data/test_set.txt')
    save_examples(dev_set, 'raw/data/dev_set.txt')
    if len(train_set) + len(test_set) + len(dev_set) != len(examples):
        raise ValueError
    return train_set, dev_set, test_set

if __name__ ==  "__main__":
    create_examples_and_sentences_files()