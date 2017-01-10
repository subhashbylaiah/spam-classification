import sys, os
import warnings
from math import log
import json
import numpy as np

class NaiveBayesSolver:

    spam_counts = {}
    notspam_counts = {}

    class_counts = {}
    model_vocabulary_size = {}
    # {
    #     spam: {total_count: 100, word_counts: {'this': {"presence_count":4, "frequency_count": 8}, 'is': {"presence_count":2, "frequency_count": 6}}},
    #     notspam: {total_count: 200, word_counts: {'YOU': {"presence_count":7, "frequency_count": 7}}}
    # }


    def build_word_counts_model(self, files_path):
        # This will build a word counts model, that will count occurence of word within in documents
        # builds the dict class_counts from the training data

        # we will be two kinds of models
        # 1) with binary features only indicating presence or absence of words in doc.
        #       So in this model, a word will only be counted once
        # 2) with continuous features. In this model, the # of times the word occurs in the doc (frequency) is counted

        print "Building word counts model..."
        dirs = os.listdir(files_path)
        for class_dir_name in dirs:
            for f in os.listdir(os.path.join(files_path, class_dir_name)):
                document = os.path.join(files_path, class_dir_name, f)
                with open(document, 'r') as file:
                    words = file.read().split()
                    distinct_words = sorted(set(words))
                    for word in distinct_words:

                        self.class_counts[class_dir_name]['word_counts'][word]['frequency_count'] = \
                            self.class_counts\
                                .setdefault(class_dir_name,{})\
                                .setdefault('word_counts',{})\
                                .setdefault(word, {})\
                                .setdefault('frequency_count', 0) + words.count(word)

                        self.class_counts[class_dir_name]['word_counts'][word]['presence_count'] = \
                            self.class_counts\
                                .setdefault(class_dir_name,{})\
                                .setdefault('word_counts',{})\
                                .setdefault(word, {})\
                                .setdefault('presence_count', 0) + 1

                    self.class_counts[class_dir_name]['total_count'] = \
                        self.class_counts\
                            .setdefault(class_dir_name, {})\
                            .setdefault('total_count', 0) + 1
                    pass
            pass

        print "#### Top 10 words most associated with spam: ####"
        spam_word_counts = self.class_counts["spam"]['word_counts']

        print "\n".join(sorted(spam_word_counts, key=spam_word_counts.get, reverse=True)[:10])
        print "##################################################"

        # To get the words that are least associated with spam, first get the notspam words
        # and find those notspamwords that are not in spamwords, and get the top of them
        notspam_word_counts = self.class_counts["notspam"]['word_counts']
        least_associated_with_spam = {k: v for k, v in notspam_word_counts.items() if k not in spam_word_counts}
        print "#### Top 10 words least associated with spam: ####"
        print "\n".join(sorted(least_associated_with_spam, key=least_associated_with_spam.get, reverse=True)[:10])
        print "##################################################"
        pass


    def train(self, files_path, model_file):
        print "Training Naive Bayes Algorithm for the given dataset..."
        self.build_word_counts_model(files_path)
        self.save_model_to_file(model_file)

    def save_model_to_file(self, file_name):
        '''
        saves the model to the file, so it can be retrieved from the file and run
        :param file_name:
        :return:
        '''
        print "Saving model to the specified file..."
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))

        with open(file_name, 'w+') as filehandle:
            json.dump({'class_counts': self.class_counts},
                      filehandle, sort_keys=True, indent=4, ensure_ascii=False, encoding="utf-8")


    def load_model_from_file(self, file_name):
        '''
        loads a previously saved model
        :param file_name:
        :return:
        '''
        with open(file_name, 'r') as filehandle:
            model = json.load(filehandle, encoding="ISO-8859-1")
            self.class_counts = model['class_counts']
        self.model_vocabulary_size = len(self.class_counts.get('spam').get('word_counts')) \
                                     + len(self.class_counts.get('notspam').get('word_counts'))


    def get_word_presence_class_log_prob(self, word, output_class):
        # Added laplace smoothing, to avoid issues due to unseen words
        # we will get some occassional warnings due to unicode decoding, which functionally does not cause any system problems,
        # hence suppressing them
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return log(float(self.class_counts[output_class]['word_counts'].get(word, {}).get('presence_count', 0) + 1.0) / \
                       (self.class_counts[output_class]['total_count'] + self.model_vocabulary_size))

    def get_word_frequency_class_log_prob(self, word, output_class):
        # Added laplace smoothing, to avoid issues due to unseen words
        # we will get some occassional warnings due to unicode decoding, which functionally does not cause any system problems,
        # hence suppressing them
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return log(float(self.class_counts[output_class]['word_counts'].get(word,{}).get('frequency_count',0) + 1.0)/\
                (self.class_counts[output_class]['total_count'] + self.model_vocabulary_size))

    def get_class_prob(self, output_class):
        total_count = 0
        for key in self.class_counts.keys():
            total_count = total_count + self.class_counts[key]['total_count']
        return log(float(self.class_counts[output_class]['total_count'])/\
            (total_count))

    def predict_simple(self, document):
        with open(document, 'r') as f:
            contents = f.read()
            words = set(contents.split())
            max_prob = -sys.maxint
            max_class = None
            for output_class in ('spam', 'notspam'):
                p = self.get_class_prob(output_class)
                for word in words:
                    p = p + self.get_word_presence_class_log_prob(word, output_class)
                if p > max_prob:
                    max_prob = p
                    max_class = output_class
                # print p
            # print "Class prediction: %s" % max_class
            return max_class

    def predict_with_word_frequencies(self, document):
        with open(document, 'r') as f:
            contents = f.read()
            words = contents.split()
            max_prob = -sys.maxint
            max_class = None
            for output_class in ('spam', 'notspam'):
                p = self.get_class_prob(output_class)
                for word in words:
                    p = p + self.get_word_frequency_class_log_prob(word, output_class)
                if p > max_prob:
                    max_prob = p
                    max_class = output_class
                    # print p
            # print "Class prediction: %s" % max_class
            return max_class


            # test_dir = '/home/subhash/Courses/AI/Assignment4/spam-classification/part1/test'
#
# dirs = os.listdir(test_dir)
# for class_dir_name in dirs:
#     total_test_cases = 0
#     correct_predictions = 0
#     for f in os.listdir(os.path.join(test_dir, class_dir_name )):
#         total_test_cases += 1
#         predicted_class = bayes_classifier.predict_simple(os.path.join(os.path.join(test_dir, class_dir_name ),f))
#         if predicted_class == class_dir_name:
#             correct_predictions += 1
#     print "Prediction Accuracy for class: %s " % class_dir_name
#     print "### Total test cases: %d,  " % (total_test_cases)
#     print "### Correct classification: %d,  " % (correct_predictions)
#     print "### Accuracy: %.2f,  " % (float(correct_predictions)/total_test_cases)

# bayes_classifier.predict_simple('/home/subhash/Courses/AI/Assignment4/spam-classification/part1/test/notspam/0002.b3120c4bcbf3101e661161ee7efcb8bf')

    def predict(self, files_path, model_file):
        self.load_model_from_file(model_file)
        # run through the tree until a leaf node is reached
        dirs = os.listdir(files_path)
        print "##### Predicting using word presence model... #####"
        for class_dir_name in dirs:
            total_test_cases = 0
            correct_predictions = 0
            for f in os.listdir(os.path.join(files_path, class_dir_name)):
                document = os.path.join(files_path, class_dir_name, f)
                total_test_cases += 1
                predicted_class = self.predict_simple(document)
                # print "Prediction for %s is %s" % (document, predicted_class)
                if predicted_class == class_dir_name:
                    correct_predictions += 1
            print "Prediction Accuracy for class: %s " % class_dir_name
            print "### Total test cases: %d,  " % (total_test_cases)
            print "### Correct classification: %d,  " % (correct_predictions)
            print "### Accuracy: %.2f,  " % (float(correct_predictions) / total_test_cases)


        print "###############################################"
        print "##### Predicting using word frequencies model... #####"
        for class_dir_name in dirs:
            total_test_cases = 0
            correct_predictions = 0
            for f in os.listdir(os.path.join(files_path, class_dir_name)):
                document = os.path.join(files_path, class_dir_name, f)
                total_test_cases += 1
                predicted_class = self.predict_with_word_frequencies(document)
                # print "Prediction for %s is %s" % (document, predicted_class)
                if predicted_class == class_dir_name:
                    correct_predictions += 1
            print "Prediction Accuracy for class: %s " % class_dir_name
            print "### Total test cases: %d,  " % (total_test_cases)
            print "### Correct classification: %d,  " % (correct_predictions)
            print "### Accuracy: %.2f,  " % (float(correct_predictions) / total_test_cases)