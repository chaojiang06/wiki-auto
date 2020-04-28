from __future__ import division

from model import *
from util import *
from datetime import timedelta
from datetime import datetime
import time
from tqdm import tqdm
from pytorch_transformers import (BertConfig,
                                  BertForSequenceClassification, BertTokenizer)
if __name__ == '__main__':

    print("start running")
    train_set = read_aligned_paragraph_data(
        "/home/chao/research_4_neural_crf_word_aligner/data/11042019_para_alignment/train_perfect_paragraph_alignment.pkl", \
        "task2", "simple_to_complex")
    dev_set = read_aligned_paragraph_data(
        "/home/chao/research_4_neural_crf_word_aligner/data/11042019_para_alignment/dev_perfect_paragraph_alignment.pkl", \
        "task2", "simple_to_complex")
    test_set = read_aligned_paragraph_data(
        "/home/chao/research_4_neural_crf_word_aligner/data/11042019_para_alignment/test_perfect_paragraph_alignment.pkl", \
        "task2", "simple_to_complex")

    dev_set_inperfect = read_inperfect_aligned_paragraph_data(
        "/home/chao/research_4_neural_crf_word_aligner/data/11042019_para_alignment/dev_inperfect_paragraph_alignment.pkl", \
        "task2", "simple_to_complex")
    test_set_inperfect = read_inperfect_aligned_paragraph_data(
        "/home/chao/research_4_neural_crf_word_aligner/data/11042019_para_alignment/test_inperfect_paragraph_alignment.pkl", \
        "task2", "simple_to_complex")

    print("starting loading data and tensor")
    all_in_one_data = read_tsv_file(
        '/home/chao/research_1_newsela_alignment_898/data/Newsela_train_dev_test_all_all_in_one_file/dev.tsv')
    print("loaded all_in_one_data")
    one_level_tensor = load_pickle_file(
        "/home/chao/research_4_neural_crf_word_aligner/data/11042019_para_alignment/output_tensor_in_level_12.pkl")
    print("loaded one_level_tensor")

    # Bert related
    MODEL_CLASSES = {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
    }
    config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']

    device = torch.device("cuda" )
    tokenizer = tokenizer_class.from_pretrained('/home/chao/research_3_pytorch-transformers/log/Newsela_train_human_dev_test_all_task2',
                                                do_lower_case=True)
    bert_for_sent_seq_model = model_class.from_pretrained('/home/chao/research_3_pytorch-transformers/log/Newsela_train_human_dev_test_all_task2', \
                                        output_hidden_states=True)
    bert_for_sent_seq_model.to(device)
    # bert_for_sent_seq_model.eval()
    # Bert related end

    sent_pait_to_cls_dict = {}
    for line_idx, line in enumerate(tqdm(all_in_one_data)):
        sent_pait_to_cls_dict[(line[6], line[5])] = one_level_tensor[line_idx]

    # aa = get_tensor_from_sent_pair(['VIRGINIA CITY, Nev. — One wonders what Mark Twain himself would make of the news: The Gold Rush-era newspaper for which he once penned stories and witticisms on frontier life as a fledgling journalist is once again in print after a decadeslong hiatus.'], \
    #                           ['VIRGINIA CITY, Nev. — One wonders what Mark Twain himself would make of the news: The Gold Rush-era newspaper for which he once wrote stories and witticisms on frontier life as a young journalist is once again in print after a decadeslong break.'], \
    #                           bert_for_sent_seq_model, tokenizer)
    #
    # print(aa)

    print('Training set size: %d' % len(train_set[0]))
    print('Developing set size: %d' % len(dev_set[0]))
    print('Testing set size: %d' % len(test_set[0]))

    num_epochs = 3

    lsents, rsents, labels, golden_sequence, paragraph_alignments = train_set
    model = NeuralWordAligner(sent_pait_to_cls_dict=sent_pait_to_cls_dict, \
                              bert_for_sent_seq_model = bert_for_sent_seq_model, \
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

        if f1 > best_dev_f1:
            best_dev_f1 = f1
            print("saving models")
            torch.save(model, "/home/chao/research_4_neural_crf_word_aligner/saved_model/best_CRF_sentence_alignment_task2_model_epoch_{}.pkl".format(epoch))

        precision, recall, f1, testset_predicted_alignment = generate_test_output_crf_sentence_alignment(test_set, model)
        print('Test score: precision: %.6f  recall: %.6f  f1: %.6f' % (precision, recall, f1))

        precision, recall, f1, testset_predicted_alignment = generate_test_output_crf_sentence_alignment_from_inperfect_paragraph_alignment(
                                                             test_set_inperfect, model)




        print('Test score: precision: %.6f  recall: %.6f  f1: %.6f' % (precision, recall, f1))

        elapsed_time = time.time() - start_time
        print('Epoch ' + str(epoch) + ' finished within ' + str(
            timedelta(seconds=elapsed_time)) + ', and current time:' + str(datetime.now()))
        model.train()
