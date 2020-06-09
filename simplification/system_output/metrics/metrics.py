import argparse
from fkgl import compute_fkgl
from sari import compute_sari


def print_metrics(complex_file, simplified_file, references_folder):

    slen, wlen, fkgl = compute_fkgl(simplified_file)
    print('Average Sentence Length : {}'.format(slen))
    print('Average Word Length : {}'.format(wlen))
    print('FKGL : {}'.format(fkgl))
    print("===============\n")

    sari_score, sarif, add, keep, deletep, deleter, deletef = compute_sari(complex_file, references_folder, simplified_file)
    print('SARI score: {}'.format(sari_score))
    print('Add : {}'.format(add))
    print('Keep : {}'.format(keep))
    print('Delete : {}'.format(deletep))
    print("===============\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--complex", dest="complex",
                        help="complex sentences", metavar="FILE")
    parser.add_argument("-r", "--reference", dest="reference",
                        help="folder that contains files with references", metavar="FILE")
    parser.add_argument("-s", "--simplified", dest="simplified",
                        help="simplified sentences", metavar="FILE")

    args = parser.parse_args()

    print_metrics(args.complex, args.simplified, args.reference)
