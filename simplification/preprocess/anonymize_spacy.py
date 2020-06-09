# -*- coding: utf-8 -*-

import spacy
import argparse
from nltk import word_tokenize


class Tagger:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg', disable=["tagger", "parser"])

    def tag(self, text):
        text = text.strip()

        sents = [s.strip() for s in text.split("<SEP>") if len(s.strip()) > 0]

        final_tagged_tokens = []
        for i, sent in enumerate(sents):
            tagged_tokens = list(self.nlp(sent))
            for tagged_token in tagged_tokens:
                final_tagged_tokens.append((tagged_token.text, tagged_token.ent_iob_, tagged_token.ent_type_))

            if i != len(sents) - 1:
                final_tagged_tokens.append(("<SEP>", "", "O"))

        return final_tagged_tokens


def create_chunks(tagged_tokens):
    ner = []
    tagged_chunks = []

    for tokens in tagged_tokens:

        token, iob, ent_type = tokens
        ent_type = "O" if len(ent_type) == 0 else ent_type

        if iob == "I":
            tagged_chunks[-1].append(token)
        else:
            tagged_chunks.append([token])
            ner.append(ent_type)

    return tagged_chunks, ner


def anonymize_entities(all_tagged_chunks, all_ner):
    sents = []
    entities = {}
    total_entities = {}

    for tagged_chunks, ner in zip(all_tagged_chunks, all_ner):
        final_text = []

        for chunk, tag in zip(tagged_chunks, ner):
            chunk = " ".join(chunk)

            if tag != "O":

                if tag not in total_entities:
                    total_entities[tag] = 0

                tag_with_number = entities.get(chunk, None)
                if tag_with_number is None:
                    total_entities[tag] += 1
                    tag_with_number = tag + "@" + str(total_entities[tag])
                    entities[chunk] = tag_with_number

                chunk = tag_with_number

            final_text.append(chunk)

        final_text = " ".join([t.strip() for t in final_text if len(t.strip()) > 0])
        sents.append(final_text)

    return sents, entities


def main(args):

    tagger = Tagger()

    fp_dict = open(args.dict, "w")
    fp_out_src = open(args.out_src, "w")
    fp_out_dst = open(args.out_dst, "w")

    n = 0
    for line1, line2 in zip(open(args.src), open(args.dst)):

        tagged_tokens1 = tagger.tag(line1.strip())
        tagged_tokens2 = tagger.tag(line2.strip())

        tagged_chunks1, ner1 = create_chunks(tagged_tokens1)
        tagged_chunks2, ner2 = create_chunks(tagged_tokens2)

        anon_text, entities = anonymize_entities([tagged_chunks1, tagged_chunks2], [ner1, ner2])

        entities = "|||".join([k + ":" + v for (k,v) in entities.items()]).strip()
        fp_out_src.write(anon_text[0] + "\n")
        fp_out_dst.write(anon_text[1] + "\n")
        fp_dict.write(entities + "\n")

        if n % 1000 == 0:
            print(n, line1.strip())
            print(n, anon_text[0])
            print(n, entities)
        n += 1

    fp_out_src.close()
    fp_out_dst.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict')
    parser.add_argument('--src')
    parser.add_argument('--dst')
    parser.add_argument('--out_src')
    parser.add_argument('--out_dst')
    args = parser.parse_args()
    main(args)
