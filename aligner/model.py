from __future__ import division
from util import *
from aligner import *
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
torch.manual_seed(1234)


class NeuralWordAligner(nn.Module):
    def __init__(self ,bert_for_sent_seq_model, tokenizer):
        super(NeuralWordAligner, self).__init__()


        self.bert_for_sent_seq_model = bert_for_sent_seq_model
        self.tokenizer = tokenizer
        max_span_size = 1
        self.max_span_size = 1

        self.mlp1 = nn.Sequential(nn.Linear(768, 768), nn.Tanh(), nn.Linear(768, 1))
        self.mlp2 = nn.Sequential(nn.Linear(6, 1))

        self.transition_matrix_dict = {}
        for len_B in range(max_span_size, 50, 1):
            extended_length_B = self.max_span_size * len_B - int(self.max_span_size * (self.max_span_size - 1) / 2)
            transition_matrix = np.zeros((extended_length_B + 1, extended_length_B + 1, 6), dtype=float)
            for j in range(extended_length_B + 1):  # 0 is NULL state
                for k in range(extended_length_B + 1):  # k is previous state, j is current state
                    if k == 0 and j == 0:
                        transition_matrix[j][k][1] = 1
                    elif k > 0 and j == 0:
                        transition_matrix[j][k][2] = 1
                    elif k == 0 and j > 0:
                        transition_matrix[j][k][3] = 1
                    # elif k<=len_B and j<=len_B:
                    # 	transition_matrix[j][k][0] = np.absolute(j - k - 1)
                    elif k > len_B and j <= len_B:
                        transition_matrix[j][k][4] = 1
                    elif k <= len_B and j > len_B:
                        transition_matrix[j][k][5] = 1
                    else:
                        transition_matrix[j][k][0] = self.distortionDistance(k, j, len_B)
            self.transition_matrix_dict[extended_length_B] = transition_matrix



    def viterbi_decoder(self, emission_matrix, transition_matrix, len_A, extended_length_B):
        """
        :param emission_matrix:  extended_length_A * (extended_length_B + 1), word/phrase pair interaction matrix
        :param transition_matrix: (extended_length_B + 1) * (extended_length_B + 1), state transition matrix
        :param len_A: source sentence length
        :param len_B: target sentence length
        :return:
        """
        emission_matrix = emission_matrix.data.cpu().numpy()
        transition_matrix = transition_matrix.data.cpu().numpy()
        T1 = np.zeros((len_A, extended_length_B + 1), dtype=float)
        T2 = np.zeros((len_A, extended_length_B + 1), dtype=int)
        T3 = np.zeros((len_A, extended_length_B + 1), dtype=int)
        for j in range(extended_length_B + 1):
            T1[0][j] = emission_matrix[0][j - 1]  # + transition_matrix[j][len_B+1]
            T2[0][j] = -1
            T3[0][j] = 1  # span size

        visited_states = set()
        for i in range(1, len_A):
            global_max_val = float("-inf")
            global_max_idx = -1
            for j in range(extended_length_B + 1):
                # if j in visited_states: # add constraint here
                # 	continue
                max_val = float("-inf")
                for span_size in range(1, min(i + 1, self.max_span_size) + 1):  # span_size can be {1,2,3,4}
                    for k in range(extended_length_B + 1):
                        if i - span_size >= 0:
                            cur_val = T1[i - span_size][k] + transition_matrix[j][k] + emission_matrix[
                                i - (span_size - 1) + (span_size - 1) * len_A - int(
                                    (span_size - 1) * (span_size - 2) / 2)][j - 1]
                        else:
                            cur_val = emission_matrix[i - (span_size - 1) + (span_size - 1) * len_A - int(
                                (span_size - 1) * (span_size - 2) / 2)][j - 1]
                        if cur_val > max_val:
                            T1[i][j] = cur_val
                            T2[i][j] = k
                            T3[i][j] = span_size
                            max_val = cur_val
                if max_val > global_max_val:
                    global_max_val = max_val
                    global_max_idx = j
        # visited_states.add(global_max_idx)
        optimal_sequence = []
        max_val = float("-inf")
        max_idx = -1
        for j in range(extended_length_B + 1):
            if T1[len_A - 1][j] > max_val:
                max_idx = j
                max_val = T1[len_A - 1][j]
        # optimal_sequence = [max_idx] + optimal_sequence
        # for i in range(len_A - 1, 0, -1):
        # 	optimal_sequence = [T2[i][max_idx]] + optimal_sequence
        # 	max_idx = T2[i][max_idx]
        i = len_A - 1
        while i >= 0:
            optimal_element = [max_idx] * T3[i][max_idx]
            optimal_sequence = optimal_element + optimal_sequence
            new_i = i - T3[i][max_idx]
            new_max_idx = T2[i][max_idx]
            i = new_i
            max_idx = new_max_idx

        return optimal_sequence

    def _score_sentence(self, output_both, transition_matrix, golden_sequence, len_A, len_B):
        # golden_sequence is a list of states: [1, 2, 3, 33, 33, 33, 8, 9, 10, 11, 41, 15]
        # print(golden_sequence)
        score = 0
        # print(output_both.size())
        gold_list = []
        tmp = golden_sequence[0:1]
        for i, item in enumerate(golden_sequence):
            if i == 0:
                continue
            if item == tmp[-1]:
                if len(tmp) == max_span_size:
                    gold_list.append((i - len(tmp), tmp, tmp[-1]))
                    tmp = [item]
                else:
                    tmp.append(item)
            else:
                gold_list.append((i - len(tmp), tmp, tmp[-1]))
                tmp = [item]
        gold_list.append((len_A - len(tmp), tmp, tmp[-1]))
        # print(gold_list)
        for start_i, span, item in gold_list:
            span_size = len(span)
            score += output_both[start_i + (span_size - 1) * len_A - int((span_size - 1) * (span_size - 2) / 2)][
                item - 1]
            if start_i - 1 >= 0:
                score += transition_matrix[item][golden_sequence[start_i - 1]]
        return score

    def argmax(self, vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return idx.item()

    def log_sum_exp(self, vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def _forward_alg(self, output_both, transition_matrix, len_A):
        target_size = output_both.size(1)
        forward_var1 = torch.full((1, target_size), 0).to(my_device)
        forward_var2 = torch.full((1, target_size), 0).to(my_device)
        forward_var3 = torch.full((1, target_size), 0).to(my_device)
        forward_var4 = torch.full((1, target_size), 0).to(my_device)
        tmp_forward_var1 = torch.full((1, target_size), 1).to(my_device)
        tmp_forward_var2 = torch.full((1, target_size), 1).to(my_device)
        tmp_forward_var3 = torch.full((1, target_size), 1).to(my_device)
        tmp_forward_var4 = torch.full((1, target_size), 1).to(my_device)
        # forward_var1 = forward_var1.to(my_device)
        # forward_var2 = forward_var2.to(my_device)
        # forward_var3 = forward_var3.to(my_device)
        # tmp_forward_var1 = tmp_forward_var1.to(my_device)
        # tmp_forward_var2 = tmp_forward_var2.to(my_device)
        # tmp_forward_var3 = tmp_forward_var3.to(my_device)
        for i in range(len_A):
            for span_size in range(1, min(i + 1, self.max_span_size) + 1):
                alphas_t = []
                if span_size == 1:
                    feat = output_both[i]
                    for j in range(target_size):
                        emit_score = feat[j - 1].view(1, -1).expand(1, target_size)
                        # print(emit_score.size())
                        # sys.exit()
                        trans_score = transition_matrix[j]  # [:-1]
                        if i <= 0:
                            next_tag_var = forward_var1 + emit_score
                        else:
                            next_tag_var = forward_var1 + trans_score + emit_score
                        # print(next_tag_var.size())
                        # print(self.log_sum_exp(next_tag_var))
                        # print(self.log_sum_exp(next_tag_var).view(1))
                        # sys.exit()
                        alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
                    tmp_forward_var1 = torch.cat(alphas_t).view(1, -1)
                elif span_size == 2 and i >= 1:
                    feat = output_both[i - 1 + len_A]
                    for j in range(target_size):
                        emit_score = feat[j - 1].view(1, -1).expand(1, target_size)
                        trans_score = transition_matrix[j]
                        if i <= 1:
                            next_tag_var = forward_var2 + emit_score
                        else:
                            next_tag_var = forward_var2 + trans_score + emit_score
                        alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
                    tmp_forward_var2 = torch.cat(alphas_t).view(1, -1)
                elif span_size == 3 and i >= 2:
                    feat = output_both[i - 2 + 2 * len_A - 1]
                    for j in range(target_size):
                        emit_score = feat[j - 1].view(1, -1).expand(1, target_size)
                        trans_score = transition_matrix[j]
                        if i <= 2:
                            next_tag_var = forward_var3 + emit_score
                        else:
                            next_tag_var = forward_var3 + trans_score + emit_score
                        alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
                    tmp_forward_var3 = torch.cat(alphas_t).view(1, -1)
                elif span_size == 4 and i >= 3:
                    feat = output_both[i - 3 + 3 * len_A - 3]
                    for j in range(target_size):
                        emit_score = feat[j - 1].view(1, -1).expand(1, target_size)
                        trans_score = transition_matrix[j]
                        if i <= 3:
                            next_tag_var = forward_var4 + emit_score
                        else:
                            next_tag_var = forward_var4 + trans_score + emit_score
                        alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
                    tmp_forward_var4 = torch.cat(alphas_t).view(1, -1)

            forward_var4 = forward_var3
            forward_var3 = forward_var2
            forward_var2 = forward_var1
            if i == 0:
                forward_var1 = tmp_forward_var1
            elif i == 1:
                max_score = torch.max(torch.max(tmp_forward_var1), torch.max(tmp_forward_var2))
                forward_var1 = max_score + torch.log(
                    torch.exp(tmp_forward_var1 - max_score) + torch.exp(tmp_forward_var2 - max_score))
            elif i >= 2:
                max_score = torch.max(torch.max(tmp_forward_var1), torch.max(tmp_forward_var2))
                max_score = torch.max(max_score, torch.max(tmp_forward_var3))
                forward_var1 = max_score + torch.log(
                    torch.exp(tmp_forward_var1 - max_score) + torch.exp(tmp_forward_var2 - max_score) + torch.exp(
                        tmp_forward_var3 - max_score))
        # elif i >= 3:
        # 	max_score = torch.max(torch.max(tmp_forward_var1), torch.max(tmp_forward_var2))
        # 	max_score = torch.max(max_score, torch.max(tmp_forward_var3))
        # 	max_score = torch.max(max_score, torch.max(tmp_forward_var4))
        # 	forward_var1 = max_score + torch.log(
        # 		torch.exp(tmp_forward_var1 - max_score) + torch.exp(tmp_forward_var2 - max_score) + torch.exp(
        # 			tmp_forward_var3 - max_score) + torch.exp( tmp_forward_var4 - max_score))

        alpha = self.log_sum_exp(forward_var1)
        return alpha


    def distortionDistance(self, state_i, state_j, sent_length):
        start_i, size_i = convert_stateID_to_spanID(state_i, sent_length)
        start_j, size_j = convert_stateID_to_spanID(state_j, sent_length)
        return np.absolute(start_j - (start_i + size_i - 1) - 1)



    def forward(self, raw_input_A, raw_input_B, golden_sequence):
        """
        :param raw_input_A: (source, source_dep_tag, source_dep_tree)
        :param raw_input_B: (target, target_dep_tag, target_dep_tree)
        :return:
        """
        # embd_A: # of chunks in A * embedding_dim

        output_type = None
        output_score = None

        syntac_loss = 0

        len_A = len(raw_input_A)
        len_B = len(raw_input_B)
        extended_length_A = len_A
        extended_length_B = len_B

        focusCube = torch.ones(len_A, len_B, 768)
        focusCube_A = torch.ones(len_A, len_B, 768)
        focusCube_B = torch.ones(len_A, len_B, 768)

        # sent_pair_cls_dict = load_pickle_file("")

        sent_A_list = []
        sent_B_list = []
        for iii in range(len_A):
            for jjj in range(len_B):
                sent_A_list.append(raw_input_A[iii])
                sent_B_list.append(raw_input_B[jjj])




        tensor_matrix = get_tensor_from_sent_pair(sent_B_list, sent_A_list , \
                        self.bert_for_sent_seq_model, self.tokenizer)



        focusCube = tensor_matrix.view(len_A, len_B, -1)



        focusCube = F.pad(focusCube, (0,0,0,1), 'constant', 0)

        focusCube = focusCube.to(my_device)
        # print(focusCube.shape)
        output_both = self.mlp1(focusCube).squeeze(2)  # extended_length_A * (extended_length_B + 1)

        # output_both = self.mlp1(focusCube).squeeze()  # extended_length_A * (extended_length_B + 1)
        pair_loss = 0



        transition_matrix = Variable(torch.from_numpy(self.transition_matrix_dict[extended_length_B])).type(
            torch.FloatTensor)
        transition_matrix = transition_matrix.to(my_device)
        # transition_matrix=self.mlp2(transition_matrix) * 0 + 1 # this is interesting
        transition_matrix = self.mlp2(transition_matrix)  # this is interesting

        transition_matrix = transition_matrix.view(transition_matrix.size(0), transition_matrix.size(1))
        if self.training:
            forward_score = self._forward_alg(output_both, transition_matrix, len_A)
            gold_score = self._score_sentence(output_both, transition_matrix, golden_sequence, len_A, len_B)
            # print(forward_score, gold_score)
            return forward_score - gold_score + syntac_loss + pair_loss * 0.1, output_type, output_score
        else:
            return_sequence = self.viterbi_decoder(output_both, transition_matrix, len_A, extended_length_B)
            return output_type, output_score, return_sequence


