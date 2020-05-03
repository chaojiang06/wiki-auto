from __future__ import division

from model import *
from util import *
from datetime import timedelta
from datetime import datetime
import time
from pytorch_transformers import (BertConfig,
                                  BertForSequenceClassification, BertTokenizer)

import argparse

def main(args):
    print("start running")
    train_set = read_aligned_paragraph_data(
        args.train_gold, \
        "task1", "simple_to_complex")
    dev_set = read_aligned_paragraph_data(
        args.dev_gold, \
        "task1", "simple_to_complex")
    test_set = read_aligned_paragraph_data(
        args.test_gold, \
        "task1", "simple_to_complex")

    dev_set_inperfect = read_inperfect_aligned_paragraph_data(
        args.dev_real, \
        "task1", "simple_to_complex")
    test_set_inperfect = read_inperfect_aligned_paragraph_data(
        args.test_real, \
        "task1", "simple_to_complex")


    # Bert related
    MODEL_CLASSES = {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
    }
    config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']

    device = torch.device("cuda" )

    tokenizer = tokenizer_class.from_pretrained(args.BERT_folder,
                                                do_lower_case=True)
    bert_for_sent_seq_model = model_class.from_pretrained(args.BERT_folder, \
                                        output_hidden_states=True)
    bert_for_sent_seq_model.to(device)
    # bert_for_sent_seq_model.eval()
    # Bert related end


    print('Training set size: %d' % len(train_set[0]))
    print('Developing set size: %d' % len(dev_set[0]))
    print('Testing set size: %d' % len(test_set[0]))

    num_epochs = 3

    lsents, rsents, labels, golden_sequence, paragraph_alignments = train_set
    model = NeuralWordAligner(bert_for_sent_seq_model = bert_for_sent_seq_model, \
                              tokenizer = tokenizer)



    model = model.to(my_device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())



    optimizer = torch.optim.Adam(parameters, \
                                 lr=2e-5, eps=1e-8)
    print('start training...')
    best_f1 = 0
    best_acc = 0
    for epoch in range(num_epochs):
        accumulated_loss = 0
        model.train()
        indices = torch.randperm(len(lsents))
        start_time = time.time()
        train_num_correct = 0
        valid_train_examples = 0
        data_loss = 0
        for index, iter_i in enumerate(indices):

            optimizer.zero_grad()

            loss, output_type, output_score = model(lsents[iter_i], rsents[iter_i], golden_sequence[iter_i])
            loss.backward()
            data_loss += loss.data
            optimizer.step()
            valid_train_examples += 1
            if valid_train_examples % 50 == 0:
                print('# ' + str(valid_train_examples) + ': Loss: ' + str(data_loss / valid_train_examples))
        print('--' * 20)
        msg = '%d completed epochs, %d batches' % (epoch, valid_train_examples)
        if valid_train_examples > 0:
            msg += '\t training batch loss: %f' % (data_loss / (valid_train_examples))
        print(msg)
        model.eval()
        best_dev_f1 = 0
        print('Results:')
        precision, recall, f1, devset_predicted_alignment = generate_test_output_crf_sentence_alignment(dev_set, model)
        print('Dev score: precision: %.6f  recall: %.6f  f1: %.6f' % (precision, recall, f1))

        precision, recall, f1, devset_predicted_alignment = generate_test_output_crf_sentence_alignment_from_inperfect_paragraph_alignment(
                                                            dev_set_inperfect, model)
        print('Dev score: precision: %.6f  recall: %.6f  f1: %.6f' % (precision, recall, f1))

        precision, recall, f1, testset_predicted_alignment = generate_test_output_crf_sentence_alignment(test_set, model)
        print('Test score: precision: %.6f  recall: %.6f  f1: %.6f' % (precision, recall, f1))

        precision, recall, f1, testset_predicted_alignment = generate_test_output_crf_sentence_alignment_from_inperfect_paragraph_alignment(
                                                             test_set_inperfect, model)




        print('Test score: precision: %.6f  recall: %.6f  f1: %.6f' % (precision, recall, f1))

        elapsed_time = time.time() - start_time
        print('Epoch ' + str(epoch) + ' finished within ' + str(
            timedelta(seconds=elapsed_time)) + ', and current time:' + str(datetime.now()))
        model.train()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--train_gold', type=str, default='', required=True, help='Path to the gold training data.')
    parser.add_argument('--dev_gold', type=str, default='', required=True, help='Path to the gold dev data.')
    parser.add_argument('--test_gold', type=str, default='', required=True, help='Path to the gold test data.')

    parser.add_argument('--dev_real', type=str, default='', required=True, help='Path to the real dev data.')
    parser.add_argument('--test_real', type=str, default='', required=True, help='Path to the real test data.')

    parser.add_argument('--BERT_folder', type=str, default='', required=True, help='Path to the fine-tuned BERT folder.')

    args = parser.parse_args()
    main(args)