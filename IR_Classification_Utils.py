import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
#from mlpython.learners.topic_modeling import DocNADE, InformationRetrieval

from internal_evaluation import load_index2label_dict, index2labels, get_shallow_label, filter_not_known_labels, evaluate
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import jsonlines
import pickle
import csv

import os, sys, random
# sys.setdefaultencoding() does not exist, here!
#reload(sys)  # Reload does the trick!
#sys.setdefaultencoding('UTF8')


#####################################################################################
#                           HELPER FUNCTIONS: BEGIN
#####################################################################################

def prepare_kernel_matrix(train_data, test_data):

    ir_kernel_matrix = np.dot(test_data, train_data.T)

    test_data_norm = np.linalg.norm(test_data, axis=1).reshape(test_data.shape[0], 1)
    train_data_norm = np.linalg.norm(train_data, axis=1)

    test_mask = (test_data_norm == 0)
    test_data_norm[test_mask] = 1.0

    train_mask = (train_data_norm == 0)
    train_data_norm[train_mask] = 1.0

    ir_kernel_matrix = ir_kernel_matrix / train_data_norm
    ir_kernel_matrix = ir_kernel_matrix / test_data_norm

    return ir_kernel_matrix

def compare_labels(train_labels, test_label, label_type="", evaluation_type=""):
    vec_goodLabel = []

    train_labels = np.asarray(train_labels)

    if label_type == "single":
        if not (isinstance(test_label, int) or isinstance(train_labels[0], int)):
            print("Labels are not instances of int")
            exit()

        test_labels = np.ones(train_labels.shape[0], dtype=np.float32) * test_label

        vec_goodLabel = np.array((train_labels == test_labels), dtype=np.int8)
    elif label_type == "multi":
        if not len(train_labels[0]) == len(test_label):
            print("Mismatched label vector length")
            exit()

        test_labels = np.asarray(test_label)
        labels_comparison_vec = np.dot(train_labels, test_labels)

        if evaluation_type == "relaxed":
            vec_goodLabel = np.array((labels_comparison_vec != 0), dtype=np.int8)

        elif evaluation_type == "strict":
            test_label_vec = np.ones(train_labels.shape[0]) * np.sum(test_label)
            vec_goodLabel = np.array((labels_comparison_vec == test_label_vec), dtype=np.int8)

        else:
            print("Invalid evaluation_type value.")

    else:
        print("Invalid label_type value.")

    return vec_goodLabel

def perform_IR_prec(kernel_matrix_test, train_labels, test_labels, list_percRetrieval=None, single_precision=False, label_type="", evaluation=""):
    '''
    :param kernel_matrix_test: shape: size = |test_samples| x |train_samples|
    :param train_labels:              size = |train_samples| or |train_samples| x num_labels
    :param test_labels:               size = |test_samples| or |test_samples| x num_labels
    :param list_percRetrieval:        list of fractions at which IR has to be calculated
    :param single_precision:          True, if only one fraction is used
    :param label_type:                "single" or "multi"
    :param evaluation:                "strict" or "relaxed", only for 
    :return:
    '''
    #print('Computing IR prec......')

    if not len(test_labels) == len(kernel_matrix_test):
        print('mismatched samples in test_labels and kernel_matrix_test')
        exit()

    prec = []

    if single_precision:
        vec_simIndexSorted = np.argsort(kernel_matrix_test, axis=1)[:, ::-1]
        prec_num_docs = np.floor(list_percRetrieval[0] * kernel_matrix_test.shape[1])
        vec_simIndexSorted_prec = vec_simIndexSorted[:, :int(prec_num_docs)]
        
        for counter, indices in enumerate(vec_simIndexSorted_prec):
            if label_type == "multi":
                classQuery = test_labels[counter, :]
                tr_labels = train_labels[indices, :]
            else:
                classQuery = test_labels[counter]
                tr_labels = train_labels[indices]
            list_percPrecision = np.zeros(len(list_percRetrieval))

            vec_goodLabel = compare_labels(tr_labels, classQuery, label_type=label_type, evaluation_type=evaluation)

            list_percPrecision[0] = np.sum(vec_goodLabel) / float(len(vec_goodLabel))

            prec += [list_percPrecision]
    else:
        vec_simIndexSorted = np.argsort(kernel_matrix_test, axis=1)[:, ::-1]
        for counter, indices in enumerate(vec_simIndexSorted):
            if label_type == "multi":
                classQuery = test_labels[counter, :]
                tr_labels = train_labels[indices, :]
            else:
                classQuery = test_labels[counter]
                tr_labels = train_labels[indices]
            list_percPrecision = np.zeros(len(list_percRetrieval))

            vec_goodLabel = compare_labels(tr_labels, classQuery, label_type=label_type, evaluation_type=evaluation)

            list_totalRetrievalCount = []
            for frac in list_percRetrieval:
                list_totalRetrievalCount.append(np.floor(frac * kernel_matrix_test.shape[1]))

            countGoodLabel = 0
            for indexRetrieval, totalRetrievalCount in enumerate(list_totalRetrievalCount):
                if indexRetrieval == 0:
                    countGoodLabel += np.sum(vec_goodLabel[:int(totalRetrievalCount)])
                else:
                    countGoodLabel += np.sum(vec_goodLabel[int(lastTotalRetrievalCount):int(totalRetrievalCount)])

                list_percPrecision[indexRetrieval] = countGoodLabel / float(totalRetrievalCount)
                lastTotalRetrievalCount = totalRetrievalCount

            prec += [list_percPrecision]  # vec_simIndexSorted[:int(list_totalRetrievalCount[0])]

    prec = np.mean(prec, axis=0)
    # print('prec:', prec)
    return prec

def get_embeddings_representation_W(data, W, doc_vector_strategy="sum", include_count=True):
    docVectors = np.zeros(shape=(len(data), W.shape[1]), dtype=np.float32)

    if include_count:
        for i, doc in enumerate(data):
            indices = doc[0][1]
            counts = doc[0][0]
            for index, count in zip(indices, counts):
                docVectors[i, :] += float(count) * W[int(index), :]
            if doc_vector_strategy == "mean" and len(indices) != 0:
                docVectors[i, :] = docVectors[i, :] / float(len(indices))
    else:
        for i, doc in enumerate(data):
            indices = doc[0][1]
            for index in indices:
                docVectors[i, :] += W[int(index), :]
            if doc_vector_strategy == "mean" and len(indices) != 0:
                docVectors[i, :] = docVectors[i, :] / float(len(indices))

    return docVectors

def get_embeddings_representation_google(data, embed_dict, index2word_dict, doc_vector_strategy="sum", include_count=True):
    docVectors = np.zeros(shape=(len(data), len(embed_dict['the'])), dtype=np.float32)

    if include_count:
        for i, doc in enumerate(data):
            indices = doc[0][1]
            counts = doc[0][0]
            for index, count in zip(indices, counts):
                word = str(index2word_dict[int(index)])
                if word in embed_dict:
                    docVectors[i, :] += float(count) * embed_dict[word]
                #else:
                #    docVectors_train[i, :] += np.random.rand(300)
            if doc_vector_strategy == "mean" and len(indices) != 0: 
                docVectors[i, :] = docVectors[i, :] / float(len(indices))
    else:
        for i, doc in enumerate(data):
            indices = doc[0][1]
            for index in indices:
                word = str(index2word_dict[int(index)])
                if word in embed_dict:
                    docVectors[i, :] += embed_dict[word]
                #else:
                #    docVectors_train[i, :] += np.random.rand(300)
            if doc_vector_strategy == "mean" and len(indices) != 0: 
                docVectors[i, :] = docVectors[i, :] / float(len(indices))

    return docVectors

def get_hidden_representation_docnade(docNadeObject, data, precision_list):
    IRObject = InformationRetrieval(docNadeObject, list_percRetrieval=precision_list)
    data_hidden, _ = IRObject.convertData(data)
    return data_hidden

def perform_classification(train_data, val_data, test_data, c_list, classification_model="logistic", norm_before_classification=False, label_type="single"):
    docVectors_train, train_labels = train_data
    docVectors_val, val_labels = val_data
    docVectors_test, test_labels = test_data

    if norm_before_classification:
        mean = np.mean(np.vstack((docVectors_train, docVectors_val, docVectors_test)), axis=0)
        std = np.std(np.vstack((docVectors_train, docVectors_val, docVectors_test)), axis=0)

        docVectors_train = (docVectors_train - mean) / std
        docVectors_val = (docVectors_val - mean) / std
        docVectors_test = (docVectors_test - mean) / std

    ## Classification Accuracy
    test_acc = []
    val_acc = []
    test_f1  = []
    val_f1  = []
    
    for c in c_list:
        if classification_model == "logistic":
            clf = LogisticRegression(C=c)
        elif classification_model == "svm":
            clf = SVC(C=c, kernel='precomputed')
        
        clf.fit(docVectors_train, train_labels)
        pred_test_labels = clf.predict(docVectors_test)
        pred_val_labels = clf.predict(docVectors_val)

        acc_test = accuracy_score(test_labels, pred_test_labels)
        acc_val = accuracy_score(val_labels, pred_val_labels)
        f1_test = precision_recall_fscore_support(test_labels, pred_test_labels, pos_label=None, average='macro')[2]
        f1_val = precision_recall_fscore_support(val_labels, pred_val_labels, pos_label=None, average='macro')[2]

        test_acc.append(acc_test)
        val_acc.append(acc_val)
        test_f1.append(f1_test)
        val_f1.append(f1_val)

    return test_acc, val_acc, test_f1, val_f1

def get_labels(docNadeObject, data, label_type="single"):
    labels = []
    prec_list = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.6, 6.4, 25.6, 100.0]

    if label_type == "single":
        for doc in data:
            label = int(doc[1])
            labels.append(label)

    elif label_type == "multi":
        IRObject = InformationRetrieval(docNadeObject, list_percRetrieval=prec_list)
        _, labels = IRObject.convertData(data)

    return np.asarray(labels)

def perform_classification_and_get_predicted_labels(train_data, val_data, test_data, c_list, classification_model="logistic", norm_before_classification=False, index2label_path="", mlb_classes=[]):
    docVectors_train, train_labels = train_data
    docVectors_val, val_labels = val_data
    docVectors_test, test_labels = test_data

    if norm_before_classification:
        mean = np.mean(np.vstack((docVectors_train, docVectors_val, docVectors_test)), axis=0)
        std = np.std(np.vstack((docVectors_train, docVectors_val, docVectors_test)), axis=0)

        docVectors_train = (docVectors_train - mean) / std
        docVectors_val = (docVectors_val - mean) / std
        docVectors_test = (docVectors_test - mean) / std

    # Results list
    match_val_results = []
    match_test_results = []

    for c in c_list:
        if classification_model == "logistic":
            clf = OneVsRestClassifier(LogisticRegression(C=c), n_jobs=5)
            #clf = OneVsRestClassifier(LinearSVC(C=c))
        elif classification_model == "svm":
            clf = OneVsRestClassifier(SVC(C=c, kernel='precomputed'))
        
        clf.fit(docVectors_train, train_labels)
        pred_test_labels = clf.predict(docVectors_test)
        pred_val_labels = clf.predict(docVectors_val)

        matches_val = evaluate(val_labels, pred_val_labels, index2label_path=index2label_path, mlb_classes=mlb_classes)
        matches_test = evaluate(test_labels, pred_test_labels, index2label_path=index2label_path, mlb_classes=mlb_classes)

        match_val_results.append(matches_val)
        match_test_results.append(matches_test)

    return match_test_results, match_val_results



def preprocess_text(text, cachedStopWords, tokenizer=None):
	if text is None: return '' 

	tokens = tokenizer.tokenize(text.lower())

	if not tokens: return ''

	filtered_tokens = []
	for token in tokens:
		if (not token in cachedStopWords) and (not token.isdigit()) and (len(str(token).strip().strip("\n").strip("\r\n")) > 1):
			#print("Token: %s len %s " %(token, len(token)))
			filtered_tokens.append(token)

	new_text = ' '.join(filtered_tokens)

	return new_text

def load_file(filename, label_key='original label'):
    docs = []
    original_labels = []

    tokenizer = RegexpTokenizer(r'\w+')
    cachedStopWords = stopwords.words("english")

    with jsonlines.open(filename) as reader:
        for i, obj in enumerate(reader):
            try:
                original_label_list = obj[label_key].split("\n")
                mapped_label = obj['mapped label']
                text  = obj['ROB']
                header = obj['header']

                # Removing digits at the beginning
                new_text = preprocess_text(text, cachedStopWords, tokenizer=tokenizer)
                new_header = preprocess_text(header, cachedStopWords, tokenizer=tokenizer)

                combined_text = new_header + ' ' + new_text

                doc_labels = []
                for l in mapped_label:
                    doc_labels.append(str(l).lower())

                if (not doc_labels) or (combined_text == ' '):
                    continue

                original_comb_text = "::" + str(header) + "  ::  " + str(text)

                docs.append(str(original_comb_text))
                original_labels.append([str(label) for label in original_label_list])
            except:
                import pdb; pdb.set_trace()
    
    return docs, original_labels

def IR_top_n_results(kernel_matrix_test, train_file_path, test_file_path, list_percRetrieval=[10], IR_file_name=""):
    '''
    :param kernel_matrix_test: shape: size = |test_samples| x |train_samples|
    :return:
    '''
    
    train_docs, train_original_labels = load_file(train_file_path, label_key='label')
    test_docs, test_original_labels = load_file(test_file_path, label_key='original label')

    fp = open(IR_file_name, "w")

    vec_simIndexSorted = np.argsort(kernel_matrix_test, axis=1)[:, ::-1]
    prec_num_docs = list_percRetrieval[0]
    vec_simIndexSorted_prec = vec_simIndexSorted[:, :int(prec_num_docs)]
    
    for counter, indices in enumerate(vec_simIndexSorted_prec):
        fp.write("\n\n")
        fp.write("Query\t::\t{" + str(test_docs[counter]) + "}\n\n")

        for index in indices:
            fp.write(str(kernel_matrix_test[counter][index]) + "\t::\t" + str(train_docs[index]) + "\n")

    fp.close()

#####################################################################################
#---------------------------Classification Report------------------------------------

def classification_report(ROB_filename, train_data, test_data, norm_before_classification=False, classification_model="logistic", \
                            c_list=[1.0], index2label_path="", rejected_labels_list=["unk"], csv_filename=""):

    docVectors_train, train_labels = train_data
    docVectors_test, test_labels = test_data

    if norm_before_classification:
        mean = np.mean(np.vstack((docVectors_train, docVectors_test)), axis=0)
        std = np.std(np.vstack((docVectors_train, docVectors_test)), axis=0)

        docVectors_train = (docVectors_train - mean) / std
        docVectors_test = (docVectors_test - mean) / std

    # Results list
    if len(c_list) > 1:
        print("More than one values of C.")
    
    for c in c_list:
        if classification_model == "logistic":
            clf = OneVsRestClassifier(LogisticRegression(C=c), n_jobs=5)
            #clf = OneVsRestClassifier(LinearSVC(C=c))
        elif classification_model == "svm":
            clf = OneVsRestClassifier(SVC(C=c, kernel='precomputed'))
        
        clf.fit(docVectors_train, train_labels)
        predicted_test_labels = clf.predict(docVectors_test)

    index2label_dict = {}
    with open(index2label_path, "rb") as f:
        index2label_dict = pickle.load(f)

    mapped_labels = index2labels(test_labels, index2label_dict)
    predicted_labels = index2labels(predicted_test_labels, index2label_dict)

    ROBs, original_labels = load_file(ROB_filename)

    csvfile = open(csv_filename, "wb")
    filewriter = csv.writer(csvfile, delimiter=',')

    for i, rob in enumerate(ROBs):
        original_label = "; ".join(original_labels[i])
        mapped_label = "; ".join([label for label in mapped_labels[i] if label not in rejected_labels_list])
        predicted_label = "; ".join([label for label in predicted_labels[i] if label not in rejected_labels_list])

        csv_file_line = [str(rob), str(original_label), str(mapped_label), str(predicted_label)]

        filewriter.writerow(csv_file_line)

    label_accuracy_count = {str(val): 0 for val in index2label_dict.values()}
    label_occurence_count = {str(val): 0 for val in index2label_dict.values()}

    for gold, pred in zip(mapped_labels, predicted_labels):
        for g in gold:
            label_occurence_count[g] += 1
            if g in pred:
                label_accuracy_count[g] += 1

    for label in label_occurence_count.keys():
        csv_file_line = [str(label), str(label_occurence_count[label]), str(label_accuracy_count[label])]
        filewriter.writerow(csv_file_line)
    
    csvfile.close()

#####################################################################################


def classification_score_without_unk_label(csv_filename, splitter=', '):
    
    csv_filename = "/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/corpora/results/" + csv_filename
    result_file_path = '.'.join(csv_filename.split('.')[:-1]) + "_without_unk.txt"
    
    ############################ WITH UNK ###########################################
    mapped_labels = []
    predictions = []

    csvfile = open(csv_filename, "rb")
    filereader = csv.reader(csvfile, delimiter=',')

    for row in filereader:
        if len(row) == 4:
            labels = [x for x in row[2].split(splitter)]
            preds = [x for x in row[3].split(splitter)]
            
            mapped_labels.append(labels)
            predictions.append(preds)

    match = evaluate(mapped_labels, predictions)

    with open(result_file_path, "a") as f:
        f.write("\n\nTest accuracy result: with UNK label\n\n")
        f.write("eval_1: " + str(match["eval_1"]) \
            + " eval_2: " + str(match["eval_2"]) \
            + " eval_3: " + str(match["eval_3"])\
            + " eval_4: " + str(match["eval_4"])\
            + " eval_5: " + str(match["eval_5"])\
            + " eval_6: " + str(match["eval_6"])\
            + " eval_7: " + str(match["eval_7"])\
            + " eval_8: " + str(match["eval_8"]))

    csvfile.close()

    ############################ WITHOUT UNK ###########################################
    mapped_labels = []
    predictions = []

    csvfile = open(csv_filename, "rb")
    filereader = csv.reader(csvfile, delimiter=',')

    for row in filereader:
        if len(row) == 4:
            labels = [x for x in row[2].split(splitter) if x != "unk"]
            preds = [x for x in row[3].split(splitter) if x != "unk"]
            
            mapped_labels.append(labels)
            predictions.append(preds)

    match = evaluate(mapped_labels, predictions)

    with open(result_file_path, "a") as f:
        f.write("\n\nTest accuracy result: without UNK label\n\n")
        f.write("eval_1: " + str(match["eval_1"]) \
            + " eval_2: " + str(match["eval_2"]) \
            + " eval_3: " + str(match["eval_3"])\
            + " eval_4: " + str(match["eval_4"])\
            + " eval_5: " + str(match["eval_5"])\
            + " eval_6: " + str(match["eval_6"])\
            + " eval_7: " + str(match["eval_7"])\
            + " eval_8: " + str(match["eval_8"]))

    csvfile.close()


def get_nearest_neighbors(filename, X, y, k, vocab, algo='ball_tree'):

    #filename = "/home/ubuntu/topic_modeling/static/rDocNADE/ROB_classification/corpora/results/" + filename

    from sklearn.neighbors import NearestNeighbors

    if np.array_equal(X, y):
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm=algo).fit(X)
        distances, indices = nbrs.kneighbors(y)

        fp = open(filename, "a")
        for i in range(len(vocab)):
            dist = distances[i][1:]
            inds = indices[i][1:]

            fp.write(str(vocab[i]) + "\t::\t")
            for i, index in enumerate(inds):
                fp.write(str(vocab[index]) + ":%.2f\t" % (dist[i]))
            fp.write("\n")
        fp.close()
    else:
        nbrs = NearestNeighbors(n_neighbors=k, algorithm=algo).fit(X)
        distances, indices = nbrs.kneighbors(y)

        fp = open(filename, "a")
        for i in range(len(vocab)):
            dist = distances[i][1:]
            inds = indices[i][1:]

            fp.write(str(vocab[i]) + "\t::\t")
            for i, index in enumerate(inds):
                fp.write(str(vocab[index]) + ":%.2f\t" % (dist[i]))
            fp.write("\n")
        fp.close()


#####################################################################################
#                           HELPER FUNCTIONS: END
#####################################################################################