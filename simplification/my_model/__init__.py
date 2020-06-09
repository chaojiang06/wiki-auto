from fairseq.models import register_model_architecture


@register_model_architecture('lstm', 'encdeca')
def my_dress(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 300)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 300)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 300)

    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 256)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 256)

    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)

    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.2)

    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.2)

    args.encoder_embed_path = getattr(args, 'encoder_embed_path', "raw_data/glove.840B.300d.txt")
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_freeze_embed = getattr(args, 'encoder_freeze_embed', False)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', True)

    args.decoder_embed_path = getattr(args, 'decoder_embed_path', "raw_data/glove.840B.300d.txt")
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_freeze_embed = getattr(args, 'decoder_freeze_embed', False)
    args.decoder_attention = getattr(args, 'decoder_attention', '1')

    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '10000,50000,200000')


@register_model_architecture('transformer', 'bert_rand')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)

    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)

    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)

    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)

    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.dropout = getattr(args, 'dropout', 0.1)

    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)

    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')

    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
