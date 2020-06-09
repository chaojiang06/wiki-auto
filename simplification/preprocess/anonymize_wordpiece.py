import argparse
from wordpiece import FullTokenizer


def main(args):
    tokenizer = FullTokenizer(args.vocab)

    fp = open(args.output, "w")
    for line in open(args.input):
        line = line.strip()
        #line = line.replace("WORK_OF_ART", "WORKOFART")
        #line = line.replace("@", "nent")
        tokens = tokenizer.tokenize(line)
        tokens.append("[SEP]")
        tokens = ["[CLS]"] + tokens
        tokenized_line = " ".join(tokens)
        tokenized_line = tokenized_line.replace("< sep >", "[SEP]")

        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            tokenized_line = tokenized_line.replace("[ " + str(i) + " ]", "[" + str(i) + "]")

        assert "\[UNK\]" not in tokenized_line
        fp.write(tokenized_line + "\n")
    fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--vocab')
    parser.add_argument('--output')
    args = parser.parse_args()
    main(args)
