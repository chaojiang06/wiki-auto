import re
import os
import torch
import tensorflow as tf


PARAMETERS_MAPPING = {

    "bert/encoder/layer_(\d+)/output/dense/kernel": ("TYPE.layers.NUM.fc2.weight",),
    "bert/encoder/layer_(\d+)/output/dense/bias": ("TYPE.layers.NUM.fc2.bias",),

    "bert/encoder/layer_(\d+)/intermediate/dense/kernel": ("TYPE.layers.NUM.fc1.weight",),
    "bert/encoder/layer_(\d+)/intermediate/dense/bias": ("TYPE.layers.NUM.fc1.bias",),

    "bert/encoder/layer_(\d+)/output/LayerNorm/gamma": ("TYPE.layers.NUM.final_layer_norm.weight",),
    "bert/encoder/layer_(\d+)/output/LayerNorm/beta": ("TYPE.layers.NUM.final_layer_norm.bias",),

    "bert/encoder/layer_(\d+)/attention/output/dense/kernel": ("TYPE.layers.NUM.self_attn.out_proj.weight",),
    "bert/encoder/layer_(\d+)/attention/output/dense/bias": ("TYPE.layers.NUM.self_attn.out_proj.bias",),

    "bert/encoder/layer_(\d+)/attention/output/LayerNorm/gamma": ("TYPE.layers.NUM.self_attn_layer_norm.weight",),
    "bert/encoder/layer_(\d+)/attention/output/LayerNorm/beta": ("TYPE.layers.NUM.self_attn_layer_norm.bias",),

    "bert/embeddings/LayerNorm/gamma": ("TYPE.layer_norm.weight",),
    "bert/embeddings/LayerNorm/beta": ("TYPE.layer_norm.bias",),

    "bert/embeddings/word_embeddings": ("TYPE.embed_tokens.weight", 4, 4 + 30522),
    "bert/embeddings/position_embeddings": ("TYPE.embed_positions.weight", 2, 2 + 512),

    "bert/encoder/layer_(\d+)/attention/self/query/kernel": ("TYPE.layers.NUM.self_attn.in_proj_weight", 0, 768),
    "bert/encoder/layer_(\d+)/attention/self/query/bias": ("TYPE.layers.NUM.self_attn.in_proj_bias", 0, 768),

    "bert/encoder/layer_(\d+)/attention/self/key/kernel": ("TYPE.layers.NUM.self_attn.in_proj_weight", 768, 2 * 768),
    "bert/encoder/layer_(\d+)/attention/self/key/bias": ("TYPE.layers.NUM.self_attn.in_proj_bias", 768, 2 * 768),

    "bert/encoder/layer_(\d+)/attention/self/value/kernel": ("TYPE.layers.NUM.self_attn.in_proj_weight", 2 * 768, 3 * 768),
    "bert/encoder/layer_(\d+)/attention/self/value/bias": ("TYPE.layers.NUM.self_attn.in_proj_bias", 2 * 768, 3 * 768)

}


def get_fairseq_path(name, mapping):
    for pattern, path in mapping.items():
        groups = re.findall(pattern, name)
        if len(groups) > 0:
            return path, groups[0]
    return None


def get_parameter(module, path, num=-1):
    tokens = path.split(".")
    for token in tokens:
        if token == "NUM":
            module = module[num]
        else:
            module = getattr(module, token)
    return module


def initialize_with_bert(bert_ckpt_path, model):

    print("Initalizing with BERT")

    tf_path = os.path.abspath(bert_ckpt_path)
    bert_init_vars = tf.train.list_variables(tf_path)

    mapping = {}
    for k, v in PARAMETERS_MAPPING.items():
        mapping[re.compile(k)] = v

    for name, shape in bert_init_vars:

        output = get_fairseq_path(name, mapping)

        if output is not None:

            array = tf.train.load_variable(tf_path, name).transpose()

            path, extra = output

            if len(path) == 1:

                path = path[0]
                layer = int(extra) if "NUM" in path else -1

                encoder = get_parameter(model, path.replace("TYPE", "encoder"), layer)
                decoder = get_parameter(model, path.replace("TYPE", "decoder"), layer)

                assert array.shape == encoder.shape and  array.shape == decoder.shape

                encoder.data = torch.from_numpy(array)
                #decoder.data = torch.from_numpy(array)

            else:
                path, start, end = path
                layer = int(extra) if "NUM" in path else -1

                encoder = get_parameter(model, path.replace("TYPE", "encoder"), layer)
                decoder = get_parameter(model, path.replace("TYPE", "decoder"), layer)

                if path.endswith("weight"):
                    if path == "TYPE.embed_tokens.weight" or path == "TYPE.embed_positions.weight":
                        array = array.transpose()
                    encoder.data[start:end, :] = torch.from_numpy(array)
                    #decoder.data[start:end, :] = torch.from_numpy(array)
                elif path.endswith("bias"):
                    encoder.data[start:end] = torch.from_numpy(array)
                    #decoder.data[start:end] = torch.from_numpy(array)

    return model


if __name__ == '__main__':
    BERT_CP_PATH = "uncased_L-12_H-768_A-12/bert_model.ckpt"
    FAIRSEQ_MODEL_PATH = "transformer.model"
    fairseq_model = torch.load(FAIRSEQ_MODEL_PATH)

    initialize_with_bert(BERT_CP_PATH, fairseq_model)
