import glob, re
import argparse


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
        alignments = [int(a.split("-")[0]) for a in alignments.split()]

        preds[number] = (hyp, alignments)
        i = i + interactive

    return preds


def main(args):

    out_anon_sents = get_output_sentences(args.out_anon, int(args.interactive), int(args.ignore_lines))

    fp = open(args.denon, "w")

    for i in range(0, len(out_anon_sents)):
        hyp, _ = out_anon_sents[i]

        if int(args.wp) == 1:
            denon_sent = hyp.replace("[SEP]", "<SEP>").strip()
            if denon_sent.endswith("<SEP>"):
                denon_sent = " ".join(denon_sent.split()[1:-1])
            else:
                denon_sent = " ".join(denon_sent.split()[1:])
            denon_sent = denon_sent.lower().strip()
            denon_sent = denon_sent.replace(" ##", "")
            denon_sent = denon_sent.replace("\' \'", '"') #'' instead of " for newsela
            denon_sent = denon_sent.replace(" - - ", " -- ")
            denon_sent = re.sub("(\d+) \, (\d+)", "\\1,\\2", denon_sent)
            denon_sent = re.sub("(\d+) \. (\d+)", "\\1.\\2", denon_sent)
            denon_sent = denon_sent.replace("- lrb -", "-lrb-")
            denon_sent = denon_sent.replace("- rrb -", "-rrb-")
            denon_sent = denon_sent.replace(" - ", "-")
            denon_sent = denon_sent.replace(" ' s ", " 's ")
            denon_sent = denon_sent.replace(" ' d ", " 'd ")
            denon_sent = denon_sent.replace(" ' m ", " 'm ")
            denon_sent = denon_sent.replace(" n ' t ", " n't ")
            denon_sent = denon_sent.replace(" ' ve ", " 've ")
            denon_sent = denon_sent.replace(" ' re ", " 're ")
            denon_sent = denon_sent.replace(" ' ll ", " 'll ")
        else:
            denon_sent = hyp.replace("@@ ", "")
        fp.write(denon_sent.lower() + "\n")

    fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_anon')
    parser.add_argument('--denon')
    parser.add_argument('--wp')
    parser.add_argument('--ignore_lines', default=5)
    parser.add_argument('--interactive', default=5)
    args = parser.parse_args()
    main(args)
