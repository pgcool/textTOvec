import argparse
import os
import csv
import numpy as np
import tensorflow as tf
import nltk
import model.data

from nltk.tokenize import RegexpTokenizer

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

csv.field_size_limit(2**28)

tokenizer = RegexpTokenizer(r'\w+')


def tokens(text):
    #return [w.lower() for w in nltk.word_tokenize(text)]
    return [w.lower() for w in tokenizer.tokenize(text)]


def counts_to_sequence(counts, ids):
    seq = []
    for i in ids:
        seq.extend([i] * int(counts[i]))
    return seq


def log_counts(ids, vocab_size):
    counts = np.bincount(ids, minlength=vocab_size)
    return np.floor(0.5 + np.log(counts + 1))


def preprocess(text, vocab_to_id, dataset_type):
    ids = [vocab_to_id.get(x) for x in tokens(text) if vocab_to_id.get(x)]
    if dataset_type == "docnade":
        counts = log_counts(ids, len(vocab_to_id))
        sequence = counts_to_sequence(counts, ids)
    else:
        sequence = ids
    return ' '.join([str(x) for x in sequence])


def main(args):
    data = model.data.Dataset(args.input)
    with open(args.vocab, 'r') as f:
        vocab = [w.strip() for w in f.readlines()]
    vocab_to_id = dict(zip(vocab, range(len(vocab))))

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    labels = {}
    for collection in data.collections:
        output_path = os.path.join(args.output, '{}.csv'.format(collection))
        #with open(output_path, 'w', newline='') as f:
        with open(output_path, 'w') as f:
            w = csv.writer(f, delimiter=',')
            for y, x in data.rows(collection, num_epochs=1):
                try:
                    if y not in labels:
                        labels[y] = len(labels)
                    w.writerow((labels[y], preprocess(x, vocab_to_id, args.dataset_type)))
                except:
                    import pdb; pdb.set_trace()

    with open(os.path.join(args.output, 'labels.txt'), 'w') as f:
        f.write('\n'.join([k for k in sorted(labels, key=labels.get)]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='path to the input dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='path to the output dataset')
    parser.add_argument('--vocab', type=str, required=True,
                        help='path to the vocab')
    parser.add_argument('--dataset-type', type=str, required=True,
                        help='dataset type to be created "docnade" or "lstm"')
    return parser.parse_args()

from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # combine label and text files
    '''
    input_text_only = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/20NS_short/N20shortTextonly.txt'
    input_label_only = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/20NS_short/N20short.LABEL'
    output_text = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/20NS_short/N20short.txt'
    train_output_text = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/20NS_short/N20short_train.txt'
    val_output_text = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/20NS_short/N20short_val.txt'
    test_output_text = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/20NS_short/N20short_test.txt'
    '''

    '''
    input = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/reuters8/r8-train_original.txt'
    output_train = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/reuters8/r8-train.txt'
    output_val = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/reuters8/r8-val.txt'

    texts = []
    labels = []
    with open(input, 'r') as f:
        for line in f.readlines():
            texts.append(line.split('\t')[1])
            labels.append(line.split('\t')[0])

    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=500, random_state=42)

    with open(output_train, 'w') as f:
        for label, text in zip(y_train, X_train):
            f.write(label+'\t'+str(text).strip('\n').replace('\n', ' ').strip()+'\n')

    with open(output_val, 'w') as f:
        for label, text in zip(y_val, X_val):
            f.write(label+'\t'+str(text).strip('\n').replace('\n', ' ').strip()+'\n')

    exit()
    '''

    '''
    # AGnews
    label_dict = {
        1: "World",
        2: "Sports",
        3: "Business",
        4: "Sci_Tech"
    }

    input_train = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/AGnewstitle/traintitletext.txt'
    train_label = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/AGnewstitle/train.LABEL'
    input_test = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/AGnewstitle/testtitletext.txt'
    test_label = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/AGnewstitle/test.LABEL'
    output_train = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/AGnewstitle/traintitle.txt'
    output_val = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/AGnewstitle/valtitle.txt'
    output_test = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/AGnewstitle/testtitle.txt'

    input_train = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/AGnews/traintext.txt'
    train_label = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/AGnews/train.LABEL'
    input_test = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/AGnews/testtext.txt'
    test_label = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/AGnews/test.LABEL'
    output_train = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/AGnews/train.txt'
    output_val = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/AGnews/val.txt'
    output_test = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/AGnews/test.txt'
    '''

    input_train = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/DBpediatopicstitle/traintitletext.txt'
    train_label = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/DBpediatopicstitle/train.LABEL'
    input_test = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/DBpediatopicstitle/testtitletext.txt'
    test_label = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/DBpediatopicstitle/test.LABEL'
    output_train = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/DBpediatopicstitle/traintitle.txt'
    output_val = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/DBpediatopicstitle/valtitle.txt'
    output_test = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/DBpediatopicstitle/testtitle.txt'

    '''
    input_train = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/DBpediatopics/traintext.txt'
    train_label = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/DBpediatopics/train.LABEL'
    input_test = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/DBpediatopics/testtext.txt'
    test_label = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/DBpediatopics/test.LABEL'
    output_train = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/DBpediatopics/train.txt'
    output_val = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/DBpediatopics/val.txt'
    output_test = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/DBpediatopics/test.txt'
    '''

    label_dict = {
        1: "Company",
        2: "EducationalInstitution",
        3: "Artist",
        4: "Athlete",
        5: "OfficeHolder",
        6: "MeanOfTransportation",
        7: "Building",
        8: "NaturalPlace",
        9: "Village",
        10: "Animal",
        11: "Plant",
        12: "Album",
        13: "Film",
        14: "WrittenWork"
    }

    with open(input_test, 'r') as f:
        texts = [line.strip('\n').replace('\t', ' ').strip() for line in f.readlines()]

    with open(test_label, 'r') as f:
        labels = [line.strip('\n').strip() for line in f.readlines()]


    with open(output_test, 'w') as f:
        for label, text in zip(labels, texts):
            f.write(label_dict[int(label)]+'\t'+str(text).strip('\n').replace('\n', '').strip()+'\n')

    with open(input_train, 'r') as f:
        texts = [line.strip('\n').replace('\t', ' ').strip() for line in f.readlines()]

    with open(train_label, 'r') as f:
        labels = [line.strip('\n').strip() for line in f.readlines()]

    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=10000, random_state=42)

    with open(output_train, 'w') as f:
        for label, text in zip(y_train, X_train):
            f.write(label_dict[int(label)]+'\t'+str(text).strip('\n').replace('\n', '').strip()+'\n')

    with open(output_val, 'w') as f:
        for label, text in zip(y_val, X_val):
            f.write(label_dict[int(label)]+'\t'+str(text).strip('\n').replace('\n', '').strip()+'\n')

    exit()

    input_text_only = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/TMNtitle/TMNtitletext.txt'
    input_label_only = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/TMNtitle/TMNtitle.LABEL'
    output_text = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/TMNtitle/TMNtitle.txt'
    train_output_text = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/TMNtitle/TMNtitle_train.txt'
    val_output_text = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/TMNtitle/TMNtitle_val.txt'
    test_output_text = '/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/DocNADE_Tensorflow/datasets/TMNtitle/TMNtitle_test.txt'

    texts = []
    with open(input_text_only, 'r') as f:
        texts = [line.strip('\n').strip() for line in f.readlines()]

    with open(input_label_only, 'r') as f:
        labels = [line.strip('\n').strip() for line in f.readlines()]

    with open(output_text, 'w') as f:
        for label, text in zip(labels, texts):
            f.write(label+'\t'+text+'\n')

    X_train, X_val_test, y_train, y_val_test = train_test_split(texts, labels, test_size = 0.30, random_state = 42)
    X_test, X_val, y_test, y_val = train_test_split(X_val_test, y_val_test, test_size = 2000, random_state = 42)

    with open(train_output_text, 'w') as f:
        for label, text in zip(y_train, X_train):
            f.write(label+'\t'+text+'\n')

    with open(val_output_text, 'w') as f:
        for label, text in zip(y_val, X_val):
            f.write(label+'\t'+text+'\n')

    with open(test_output_text, 'w') as f:
        for label, text in zip(y_test, X_test):
            f.write(label+'\t'+text+'\n')

    exit()
    main(parse_args())
