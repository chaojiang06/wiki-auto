from __future__ import division
from aligner import *
import random
import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)
import pickle

seed = 123456
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
max_span_size = 1

my_device = torch.device('cuda:0')
torch.cuda.manual_seed_all(seed)

def convert_stateID_to_spanID(stateID, sent_length):  # 0 is NULL state
    stateID = stateID - 1
    if stateID < 0:
        return (-1, -1)
    else:
        for span_length in range(1, max_span_size + 1):
            lower_bound = (span_length - 1) * sent_length - int((span_length - 1) * (span_length - 2) / 2)
            upper_bound = span_length * sent_length - int(span_length * (span_length - 1) / 2)
            if stateID >= lower_bound and stateID < upper_bound:
                return (stateID - lower_bound, span_length)  # return (spanID, span_Length)


def load_pickle_file(path):
    with open(path, 'rb') as handle:
        b = pickle.load(handle)

    return b


def save_as_pickle_file(file, path):
    with open(path, 'wb') as handle:
        pickle.dump(file, handle, protocol=2)


def read_tsv_file(path):
    data = []

    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quotechar='', quoting=csv.QUOTE_NONE)
        for idx, line in enumerate(reader):
            #             line = [entry.decode("utf8") for entry in line]
            data.append(line)

    return data


def read_aligned_paragraph_data(path, task1_or_task2, simple_to_complex_or_reverse):
    paragraph_alignments = load_pickle_file(path)
    lsents = []
    rsents = []
    labels = []
    golden_sequences = []

    for paragraph_alignment in paragraph_alignments:
        lsents.append(paragraph_alignment['simple_side_sents'])
        rsents.append(paragraph_alignment['complex_side_sents'])
        if task1_or_task2 == 'task1':
            labels.append(paragraph_alignment['alignment_in_new_format'] + \
                          paragraph_alignment['partial_alignment_in_new_format'])
        else:
            labels.append(paragraph_alignment['alignment_in_new_format'])

    if simple_to_complex_or_reverse == "simple_to_complex":

        for label_idx, label in enumerate(labels):
            labels[label_idx] = [[i[1], i[0]] for i in label]

    if simple_to_complex_or_reverse == "complex_to_simple":
        lsents, rsents = rsents, lsents

    for label_idx, label in enumerate(labels):
        matching_dict = {}
        sub_gold_sequence = []
        for a_pair_of_label in label:
            a, b = a_pair_of_label
            a = int(a.split("_")[-1])
            b = int(b.split("_")[-1])

            if a in matching_dict:
                matching_dict[a].append(b)
            else:
                matching_dict[a] = [b]

        for i in range(len(lsents[label_idx])):

            if i in matching_dict:
                sub_gold_sequence.append(min(matching_dict[i]) + 1)
            else:
                sub_gold_sequence.append(0)

        golden_sequences.append(sub_gold_sequence)

    return lsents, rsents, labels, golden_sequences, paragraph_alignments


def read_inperfect_aligned_paragraph_data(path, task1_or_task2, simple_to_complex_or_reverse):
    paragraph_alignments, perfect_sent_alignment_list = load_pickle_file(path)
    lsents = []
    rsents = []
    labels = []
    golden_sequences = []

    for paragraph_alignment in paragraph_alignments:
        lsents.append(paragraph_alignment['simple_side_sents'])
        rsents.append(paragraph_alignment['complex_side_sents'])
        if task1_or_task2 == 'task1':
            labels.append(paragraph_alignment['alignment_in_new_format'] + \
                          paragraph_alignment['partial_alignment_in_new_format'])
        else:
            labels.append(paragraph_alignment['alignment_in_new_format'])

    if simple_to_complex_or_reverse == "simple_to_complex":

        for label_idx, label in enumerate(labels):
            labels[label_idx] = [[i[1], i[0]] for i in label]

    if simple_to_complex_or_reverse == "complex_to_simple":
        lsents, rsents = rsents, lsents

    for label_idx, label in enumerate(labels):
        matching_dict = {}
        sub_gold_sequence = []
        for a_pair_of_label in label:
            a, b = a_pair_of_label
            a = int(a.split("_")[-1])
            b = int(b.split("_")[-1])

            if a in matching_dict:
                matching_dict[a].append(b)
            else:
                matching_dict[a] = [b]

        for i in range(len(lsents[label_idx])):

            if i in matching_dict:
                sub_gold_sequence.append(min(matching_dict[i]) + 1)
            else:
                sub_gold_sequence.append(0)

        golden_sequences.append(sub_gold_sequence)

    return lsents, rsents, labels, golden_sequences, paragraph_alignments, perfect_sent_alignment_list


def generate_test_output_crf_sentence_alignment_from_inperfect_paragraph_alignment(test_set, model):
    test_lsents, test_rsents, _, _, paragraph_alignments, perfect_sent_alignment = test_set

    predicted_alignment = []

    for test_i in range(len(test_lsents)):

        output_type, output_score, predect_sequence = model(test_lsents[test_i], test_rsents[test_i], None)

        # print(test_lsents[test_i])
        # print(test_rsents[test_i])
        # print(predect_sequence)
        # print(golden_sequence[test_i])

        sub_prediected_alignment = []

        for i in range(len(predect_sequence)):

            if predect_sequence[i] != 0:
                small_pair = ["simple_{}".format(i), "complex_{}".format(predect_sequence[i] - 1)]
                sub_prediected_alignment.append([paragraph_alignments[test_i]["new_to_original"][small_pair[0]], \
                                                 paragraph_alignments[test_i]["new_to_original"][small_pair[1]]])

        predicted_alignment.extend(sub_prediected_alignment)

    tmptmp_predicted_alignment = [[i[1], i[0]] for i in predicted_alignment]
    tp_count = len([i for i in tmptmp_predicted_alignment if i in perfect_sent_alignment])
    fp_count = len(predicted_alignment) - tp_count
    total_count = len(perfect_sent_alignment)
    fn_count = total_count - tp_count

    precision = tp_count * 1.0 / (tp_count + fp_count)
    recall = tp_count * 1.0 / total_count
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    # upp_bound = 0
    # for i in golden_sequence:
    #     upp_bound += len([jj for jj in i if jj != 0])
    #
    # print("in this dataset, the total positive that could be predicted is {}".format(upp_bound))

    print("tp {}, fp {}, fn {}, total positive {}".format(tp_count, fp_count, fn_count, total_count))

    # print(tmptmp_predicted_alignment)
    # print(perfect_sent_alignment)

    return precision, recall, f1, predicted_alignment


def generate_test_output_crf_sentence_alignment(test_set, model):
    test_lsents, test_rsents, test_labels, golden_sequence, paragraph_alignments = test_set

    predicted_alignment = []

    tmptmp_gold_alignments = []

    for i in paragraph_alignments:
        tmptmp_gold_alignments.extend(i['alignment'])
        tmptmp_gold_alignments.extend(i['partial_alignment'])

    for test_i in range(len(test_labels)):

        output_type, output_score, predect_sequence = model(test_lsents[test_i], test_rsents[test_i], None)

        for i in range(len(predect_sequence)):

            if predect_sequence[i] != 0:
                small_pair = ["simple_{}".format(i), "complex_{}".format(predect_sequence[i] - 1)]
                predicted_alignment.append([paragraph_alignments[test_i]["new_to_original"][small_pair[0]], \
                                            paragraph_alignments[test_i]["new_to_original"][small_pair[1]]])

    tmptmp_predicted_alignment = [[i[1], i[0]] for i in predicted_alignment]

    tp_count = len([i for i in tmptmp_predicted_alignment if i in tmptmp_gold_alignments])
    fp_count = len(predicted_alignment) - tp_count
    total_count = len(tmptmp_gold_alignments)
    fn_count = total_count - tp_count

    upp_bound = 0
    for i in golden_sequence:
        upp_bound += len([jj for jj in i if jj != 0])

    print("in this dataset, the total positive that could be predicted is {}".format(upp_bound))

    print("tp {}, fp {}, fn {}, total positive {}".format(tp_count, fp_count, fn_count, total_count))
    precision = tp_count * 1.0 / (tp_count + fp_count)
    recall = tp_count * 1.0 / total_count
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1, predicted_alignment

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """


    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):


        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)


        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask  # differentiate which part is input, which part is padding
        self.segment_ids = segment_ids  # differentiate different sentences
        self.label_id = label_id

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_tensor_from_sent_pair(sentA, sentB, model, tokenizer, mode = 'train'):
    # fake_example = [InputExample(guid=111, text_a=sentA, text_b=sentB, label=None)]
    model.eval()
    fake_example = []
    for i in range(len(sentA)):
        fake_example.append(InputExample(guid=i, text_a=sentA[i], text_b=sentB[i], label='good'))


    fake_example_features = convert_examples_to_features(fake_example, ["good", 'bad'], 128, tokenizer, 'classification',
                                                         cls_token_at_end=bool('bert' in ['xlnet']),
                                                         # xlnet has a cls token at the end
                                                         cls_token=tokenizer.cls_token,
                                                         cls_token_segment_id=2 if 'bert' in ['xlnet'] else 0,
                                                         sep_token=tokenizer.sep_token,
                                                         sep_token_extra=bool('bert' in ['roberta']),
                                                         # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                         pad_on_left=bool('bert' in ['xlnet']),
                                                         # pad on the left for xlnet
                                                         pad_token=
                                                         tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                         pad_token_segment_id=4 if 'bert' in ['xlnet'] else 0,
                                                         )

    all_input_ids = torch.tensor([f.input_ids for f in fake_example_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in fake_example_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in fake_example_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in fake_example_features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # # batch = dataset[0]
    # # batch = tuple(t.to(my_device) for t in batch)
    #
    # # model.eval()
    #
    # inputs = {'input_ids': torch.stack([i[0] for i in dataset]).to(my_device),
    #           'attention_mask': torch.stack([i[1] for i in dataset]).to(my_device),
    #           'token_type_ids': torch.stack([i[2] for i in dataset]).to(my_device),
    #           # XLM and RoBERTa don't use segment_ids
    #           'labels': torch.stack([i[3] for i in dataset]).to(my_device)}
    #
    # outputs = model(input_ids=inputs["input_ids"], \
    #                 attention_mask=inputs["attention_mask"], \
    #                 token_type_ids=inputs["token_type_ids"], \
    #                 labels=None, \
    #                 )
    #
    # # outputs = outputs.data()
    # outputs = outputs[1][-1][:, 0, :]
    # outputs = outputs.data
    output_tensor = []
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=8)
    for batch in eval_dataloader:
        batch = tuple(t.to(my_device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(input_ids=inputs["input_ids"], \
                            attention_mask=inputs["attention_mask"], \
                            token_type_ids=inputs["token_type_ids"], \
                            labels=None, \
                            )

            output_tensor.append(outputs[-1][-1][:, 0, :].data)

    output_tensor = torch.cat(output_tensor)

    return output_tensor



