import sys, os
from math import log
import json
import numpy as np
import warnings

class Node(object):
    """Class to represent the Data structure needed to represent a node in the tree"""

    def __init__(self, right_tree=None, left_tree=None, isleaf=None, attribute=None, attribute_idx=None, attribute_split_val=None,
                 positive_class_count=None, negative_class_count=None, depth=0):
        self.right_tree = right_tree  # pointer to the right subtree
        self.left_tree = left_tree  # pointer to the left subtree
        self.isleaf = isleaf  # value of the output class if it is the leaf node, None if not
        self.attribute = attribute # attribute(word) to split on
        self.attribute_idx = attribute_idx  # Column number of the attribute based on which the tree is split
        self.attribute_split_val = attribute_split_val  # Value of the attribute based on which its split
        self.positive_class_count = positive_class_count # positive class count at this node
        self.negative_class_count = negative_class_count # negative class count at this node
        self.depth = depth

class DecisionTreeSolver:
    '''
    Class that implements the decision tree algorithm
    '''
    # lets have a stop word list and have the implementation needed to process a list of stop words, remove them
    # email messages are cluttered with a lot of stop words and this can be expanded
    stopwords = ['|', '&nbsp;<a', '-->', '<!--']

    # dictionary for class labels
    class_labels = {'spam':1, 'notspam':0}

    def __init__(self, max_depth=5):
        self.doc_vectors_binary = None
        self.doc_vectors_frequencies = None
        self.features = None
        self.max_depth = max_depth
        self.tree = None
        self.class_counts = {}
        # Eg. class_counts dictionary
        # {
        #     spam: {total_count: 100, word_counts: {'this':4, 'is': 4, 'test': 4}},
        #     notspam: {total_count: 200, word_counts: {'YOU':4, 'ARE': 4, 'SELECTED': 4}},
        # }


    def compute_entropy(self, data_array):
        '''
        Function to compute the entropy
        :param data_array: output class labels array
        :return: entropy value
        '''
        entropy_val = 0
        for c in (self.class_labels['spam'], self.class_labels['notspam']):
            p_x = float(len(data_array[data_array[:,-1] == c])) / len(data_array)
            if p_x != 0:
                entropy_val -= p_x * log(p_x, 2)  # Calculating the entropy
        return entropy_val



    def build_word_counts_model(self, files_path):
        '''
        This will build a word counts model, that will count occurence of word within in documents
        builds the dict class_counts from the training data
        :param files_path: path to the dir with the training files
        :return: Updates the class_counts dictionary for the object
        '''
        dirs = os.listdir(files_path)
        for class_dir_name in dirs:
            for f in os.listdir(os.path.join(files_path, class_dir_name)):
                document = os.path.join(files_path, class_dir_name, f)
                with open(document, 'r') as file:
                    contents = file.read()
                    words = set(contents.split())
                    for word in words:
                        if word not in self.stopwords:
                            self.class_counts[class_dir_name]['word_counts'][word] = \
                                self.class_counts\
                                    .setdefault(class_dir_name,{})\
                                    .setdefault('word_counts',{})\
                                    .setdefault(word, 0) + 1

                    self.class_counts[class_dir_name]['total_count'] = \
                        self.class_counts\
                            .setdefault(class_dir_name, {})\
                            .setdefault('total_count', 0) + 1
                    pass
            pass
        pass


    def build_doc_to_words_vector(self, num_top_words, files_path):
        '''
        Function to build the feature vectors
        :param num_top_words: Number of top words based on which to build the feature vectors
                    This is used to grab that many words from both spam and notspam word count lists
        :param files_path: path to training corpus
        :return:
        '''

        # build the top_words_list from spam and notspam dicts
        spam_word_counts = self.class_counts.get('spam').get('word_counts')
        notspam_word_counts = self.class_counts.get('notspam').get('word_counts')

        num_spam_words = len(spam_word_counts) if len(spam_word_counts) <  num_top_words else num_top_words
        num_notspam_words = len(notspam_word_counts) if len(notspam_word_counts) <  num_top_words else num_top_words

        top_spam_words_list = sorted(spam_word_counts, key=spam_word_counts.get, reverse=True)[:num_spam_words]
        top_notspam_words_list = sorted(notspam_word_counts, key=notspam_word_counts.get, reverse=True)[:num_notspam_words]

        top_words_list = set(top_spam_words_list + top_notspam_words_list)
        top_words_list = list(top_words_list)
        self.features = top_words_list

        # initialize an empty feature matrix
        self.doc_vectors_binary = np.empty((0, len(top_words_list) + 1), dtype=int) #feature vector for binary features
        self.doc_vectors_frequencies = np.empty((0, len(top_words_list) + 1), dtype=int) # feature vector for frequencies

        # process the input files and build the feature matrix
        dirs = os.listdir(files_path)
        for class_dir_name in dirs:
            i = 0
            print "Processing files in " + class_dir_name + " folder..."

            for f in os.listdir(os.path.join(files_path, class_dir_name)):
                document = os.path.join(files_path, class_dir_name, f)
                with open(document, 'r') as file:
                    contents = file.read()

                    words1 = set(contents.split())
                    bin_vec = map(lambda x: 1 if x else 0,[top_word in words1 for top_word in top_words_list])
                    self.doc_vectors_binary = np.append(self.doc_vectors_binary, np.reshape(bin_vec  + [self.class_labels[class_dir_name]],(1, len(bin_vec)+1)), axis=0)

                    words2 = contents.split()
                    freq_vec = [words2.count(top_word) for top_word in top_words_list]
                    self.doc_vectors_frequencies = np.append(self.doc_vectors_frequencies, np.reshape(freq_vec + [self.class_labels[class_dir_name]],(1, len(freq_vec)+1)), axis=0)

                # i += 1
                # print i
        # docs_with_no_words_from_list = self.doc_vectors[np.sum(np.asarray(self.doc_vectors[:, :-5], int), axis=4) == 0]
        pass



    def get_majority_class(self, labels):
        # returns the majority class for the labels array
        majority_class = None
        majority_count = -sys.maxint
        for value in (self.class_labels['spam'], self.class_labels['notspam']):
            count = len(labels[labels == value])
            if count >  majority_count:
                majority_class = value
                majority_count = count
        return majority_class, majority_count


    def decision_tree_learner(self, data_array, indices, parent_major_class, depth):
        # function that implemets the decision tree learner recursive algorithm
        # follows the pseudo code provided by Steve and Russell AI book
        if len(data_array) == 0:
            return Node(isleaf=parent_major_class, positive_class_count=0, negative_class_count=0, depth=depth)

        majority_class, majority_count = self.get_majority_class(data_array[:, -1])
        positive_class_count = len(data_array[data_array[:, -1] == self.class_labels['spam']])
        negative_class_count = len(data_array[data_array[:, -1] == self.class_labels['notspam']])

        if majority_count == len(data_array):
            return Node(isleaf=majority_class, positive_class_count=positive_class_count, negative_class_count=negative_class_count, depth=depth)

        if len(indices) == 0:
            return Node(isleaf=majority_class, positive_class_count=positive_class_count, negative_class_count=negative_class_count, depth=depth)

        # if max_depth is reached return node, stop building tree
        if depth == self.max_depth:
            return Node(isleaf=majority_class, positive_class_count=positive_class_count, negative_class_count=negative_class_count, depth=depth)

        # Find the best attribute to split on
        lowest_entropy = sys.maxint
        best_attr_index = None
        best_attr = None

        # loop over all word features except last (last col has class output)
        for column in indices:

            unique_values = set(data_array[:, column])  # Set of al the unique values in the present column
            for value in unique_values:
                cum_entropy = 0
                # filter the data based on the feature values and compute entropy
                # for value in ('0','1'):
                # Split the column based on the value
                left_branch = data_array[data_array[:, column] >= value]
                right_branch = data_array[data_array[:, column] < value]

                # filtered = data_array[data_array[:, column] == value]
                if len(left_branch) > 0 and len(right_branch) > 0:
                    x1 = float(left_branch.shape[0]) / (data_array.shape[0])
                    cum_entropy += x1 * self.compute_entropy(left_branch[:, [-1]])

                # if len(right_branch) > 0:
                    x1 = float(right_branch.shape[0]) / (data_array.shape[0])
                    cum_entropy += x1 * self.compute_entropy(right_branch[:, [-1]])

                    # Check if the entropy is lower than the lowest
                    if cum_entropy < lowest_entropy:
                        lowest_entropy = cum_entropy
                        best_attr_index = column
                        best_attr = self.features[column]
                        best_split_val = value

        # print "Splitting feature %s at value %s that has entropy %f " % (self.features[best_attr_index], str(best_split_val), lowest_entropy)
        tree = Node(attribute=self.features[best_attr_index], attribute_idx=best_attr_index,
                    attribute_split_val=best_split_val,
                    positive_class_count=positive_class_count, negative_class_count=negative_class_count, depth=depth)
        majority_class, majority_count = self.get_majority_class(data_array[:, -1])

        indices.remove(best_attr_index)
        left_branch = data_array[data_array[:, best_attr_index] >= best_split_val]
        left_tree = self.decision_tree_learner(left_branch, indices, majority_class, depth+1)
        tree.left_tree = left_tree

        right_branch = data_array[data_array[:, best_attr_index] < best_split_val]
        right_tree = self.decision_tree_learner(right_branch, indices, majority_class, depth+1)
        tree.right_tree = right_tree

        return tree

    def print_tree(self, tree, side=None):
        print_string = ""
        if side == "Left":
            print_string = "(value == True )"
        if side == "Right":
            print_string = "(value == False)"
        if tree.isleaf is None:
            print "=====" * tree.depth +  print_string + " :: Split on    ::  " + tree.attribute + " >= " + str(tree.attribute_split_val)
            self.print_tree(tree.left_tree, side="Left")
            self.print_tree(tree.right_tree, side="Right")
        elif tree.isleaf is not None:
            print "=====" * tree.depth +  print_string + " :: Classify as ::  " + str(tree.isleaf)


    def convert_tree_to_json(self, tree):
        if tree.isleaf is not None:
            return {
                    "right_tree": tree.right_tree,
                    "left_tree": tree.left_tree,
                    "isleaf": tree.isleaf,
                    "attribute": tree.attribute,
                    "attribute_idx": tree.attribute_idx,
                    "attribute_split_val": tree.attribute_split_val,
                    "positive_class_count": tree.positive_class_count,
                    "negative_class_count":  tree.negative_class_count,
                    "depth": tree.depth
                }
        elif tree.isleaf is None:
            left_tree = self.convert_tree_to_json(tree.left_tree)
            right_tree = self.convert_tree_to_json(tree.right_tree)
            return {
                    "right_tree": right_tree,
                    "left_tree": left_tree,
                    "isleaf": tree.isleaf,
                    "attribute": tree.attribute,
                    "attribute_idx": tree.attribute_idx,
                    "attribute_split_val": tree.attribute_split_val,
                    "positive_class_count": tree.positive_class_count,
                    "negative_class_count":  tree.negative_class_count,
                    "depth": tree.depth
                }


    def train(self, files_path, model_file):
        print "Training Decision Tree Algorithm for the given dataset..."
        self.build_word_counts_model(files_path)
        self.build_doc_to_words_vector(1000, files_path)

        print "Training a model using binary features..."
        tree_binary = self.decision_tree_learner(data_array=self.doc_vectors_binary, indices=range(self.doc_vectors_binary.shape[1]-1), parent_major_class=None, depth=1)

        print "Training a model using frequency features..."
        tree_freq = self.decision_tree_learner(data_array=self.doc_vectors_frequencies, indices=range(self.doc_vectors_frequencies.shape[1]-1), parent_major_class=None, depth=1)


        print "Decision Tree learnt using binary features..."
        print "#####################################################"
        self.print_tree(tree_binary)
        print "#####################################################"

        print "Decision Tree learnt using Frequency based features..."
        print "#####################################################"
        self.print_tree(tree_freq)
        print "#####################################################"

        self.save_model_to_file(model_file, tree_binary, tree_freq)
        pass
        # self.save_model_to_file(self.model_file)


    def save_model_to_file(self, file_name, tree_binary, tree_freq):
        '''
        saves the model to the file, so it can be retrieved from the file and run
        :param file_name:
        :return:
        '''
        print "Saving the models to the specified file..."
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))

        json_tree_bin = self.convert_tree_to_json(tree_binary)
        json_tree_freq = self.convert_tree_to_json(tree_freq)
        with open(file_name, 'w+') as filehandle:
            json.dump({"bin_tree":json_tree_bin, "freq_tree":json_tree_freq}, filehandle, sort_keys = True, indent = 4, ensure_ascii = False, encoding = "utf-8")

            # json.dump({'class_counts': self.class_counts},
            #           filehandle, sort_keys=True, indent=4, ensure_ascii=False, encoding="utf-8")


    def convert_json_to_tree(self, json_tree):
        # recursive algorithm to convert the model read from the file back into the Tree data structure used at prediction
        if json_tree.get("isleaf") in [0,1]:
            return Node(
                right_tree = json_tree.get("right_tree"),
                left_tree = json_tree.get("left_tree"),
                isleaf = json_tree.get("isleaf"),
                attribute = json_tree.get("attribute"),
                attribute_idx = json_tree.get("attribute_idx"),
                attribute_split_val = json_tree.get("attribute_split_val"),
                positive_class_count = json_tree.get("positive_class_count"),
                negative_class_count = json_tree.get("negative_class_count"),
                depth = json_tree.get("depth")
            )
        else:
            left_tree = self.convert_json_to_tree(json_tree.get("left_tree"))
            right_tree = self.convert_json_to_tree(json_tree.get("right_tree"))
            return Node(
                right_tree = right_tree,
                left_tree = left_tree,
                isleaf = json_tree.get("isleaf"),
                attribute = json_tree.get("attribute"),
                attribute_idx = json_tree.get("attribute_idx"),
                attribute_split_val = json_tree.get("attribute_split_val"),
                positive_class_count = json_tree.get("positive_class_count"),
                negative_class_count = json_tree.get("negative_class_count"),
                depth = json_tree.get("depth")
            )

    def load_model_from_file(self, file_name):
        '''
        loads a previously saved model
        :param file_name:
        :return:
        '''
        with open(file_name, 'r') as filehandle:
            json_tree = json.load(filehandle)

        tree_bin = self.convert_json_to_tree(json_tree["bin_tree"])
        tree_freq = self.convert_json_to_tree(json_tree["freq_tree"])

        return (tree_bin, tree_freq)

    def predict_for_input(self, tree, word_vector):
        # run through the tree until a leaf node is reached
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            while tree.isleaf is None:
                if word_vector.count(tree.attribute) >= tree.attribute_split_val:
                    tree = tree.left_tree
                else:
                    tree = tree.right_tree
            return tree.isleaf

    def predict(self, files_path, model_file):
        print "Predicting using Decision Tree Algorithm for the given test set..."

        (tree_bin, tree_freq) = self.load_model_from_file(model_file)


        dirs = os.listdir(files_path)

        print "Predicting using binary features model..."
        print "#############################################################"
        for class_dir_name in dirs:
            total_test_cases = 0
            correct_predictions = 0
            for f in os.listdir(os.path.join(files_path, class_dir_name)):
                document = os.path.join(files_path, class_dir_name, f)
                total_test_cases += 1
                with open(document, 'r') as file:
                    contents = file.read()
                    words = set(contents.split())
                    words = list(words)
                    predicted_class = self.predict_for_input(tree_bin, words)
                    # print "Prediction for %s is %s" % (document, predicted_class)
                    if predicted_class == self.class_labels[class_dir_name]:
                        correct_predictions += 1
            print "Prediction Accuracy for class: %s " % class_dir_name
            print "### Total test cases: %d,  " % (total_test_cases)
            print "### Correct classification: %d,  " % (correct_predictions)
            print "### Accuracy: %.2f,  " % (float(correct_predictions)/total_test_cases)
        print "#############################################################"


        print "Predicting using frequency based features model..."
        print "#############################################################"
        for class_dir_name in dirs:
            total_test_cases = 0
            correct_predictions = 0
            for f in os.listdir(os.path.join(files_path, class_dir_name)):
                document = os.path.join(files_path, class_dir_name, f)
                total_test_cases += 1
                with open(document, 'r') as file:
                    contents = file.read()
                    words = contents.split()

                    predicted_class = self.predict_for_input(tree_freq, words)
                    # print "Prediction for %s is %s" % (document, predicted_class)
                    if predicted_class == self.class_labels[class_dir_name]:
                        correct_predictions += 1
            print "Prediction Accuracy for class: %s " % class_dir_name
            print "### Total test cases: %d,  " % (total_test_cases)
            print "### Correct classification: %d,  " % (correct_predictions)
            print "### Accuracy: %.2f,  " % (float(correct_predictions) / total_test_cases)
        print "#############################################################"

