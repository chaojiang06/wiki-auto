import glob
import argparse


def get_entities_from_dictionary(fname):
    token_delimiter = "|||"
    ne_delimiter = ":"

    all_entities = []
    for line in open(fname):
        entities = {}
        text = line.strip()

        if len(text) > 0:
            for word_ne in text.split(token_delimiter):
                tokens = word_ne.split(ne_delimiter)
                ne = tokens[-1]
                word = ":".join(tokens[:-1])
                entities[ne] = word

        all_entities.append(entities)

    return all_entities


def get_output_sentences(file_name, interactive=5, lines_to_ignore=5):

    if interactive == 4:
        lines = open(file_name).readlines()[lines_to_ignore:]
    else:
        lines = open(file_name).readlines()[lines_to_ignore:-2]

    i = 0
    preds = {}
    while i < len(lines):

        token = lines[i].split()[0]
        number = int(token.split("-")[1])

        hyp = lines[i + (interactive - 3)].split("\t")[2].strip()
        alignments = lines[i + (interactive - 1)].split("\t")[1].strip()
        alignments = [int(i) for i in alignments.split()]

        preds[number] = (hyp.split(), alignments)
        i = i + interactive

    return preds


def replace_unk(hyp, alignments, org_sent, entities):

    org_sent = org_sent.split()
    org_sent.append("")

    new_hyp = []
    for word, a in zip(hyp, alignments):

        if word == "<unk>":
            word = org_sent[a]

        if "@" in word and word not in entities:

            if a in org_sent and (org_sent[a] != "." or org_sent[a] != ","):
                word = org_sent[a]
                if org_sent[a] in entities:
                    word = entities[org_sent[a]]

        new_hyp.append(word)

    return new_hyp


def main(args):

    entities = get_entities_from_dictionary(args.dict)
    org_sents = [line.strip() for line in open(args.src_anon)]
    out_anon_sents = get_output_sentences(args.out_anon, int(args.interactive), int(args.ignore_lines))

    fp = open(args.denon, "w")

    for i in range(0, len(out_anon_sents)):

        ent = entities[i]

        hyp, alignments = out_anon_sents[i]
        hyp_no_unk = replace_unk(hyp, alignments, org_sents[i], ent)

        denon_sent = []
        for word in hyp_no_unk:
            if word in ent:
                word = ent[word]
            denon_sent.append(word)

        denon_sent = " ".join(denon_sent).strip().lower()
        fp.write(denon_sent + "\n")
        # fp.write(" ".join(denon_sent).strip() + "\n")

    fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict', default=None)
    parser.add_argument('--src')
    parser.add_argument('--src_anon')
    parser.add_argument('--dst')
    parser.add_argument('--dst_anon')
    parser.add_argument('--out_anon')
    parser.add_argument('--denon')
    parser.add_argument('--ignore_lines', default=5)
    parser.add_argument('--interactive', default=5)
    args = parser.parse_args()
    main(args)