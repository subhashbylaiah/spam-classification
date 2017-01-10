# Spam detection using Naive Bayes and Decision trees

import DecisionTreeSolver
import NaiveBayesSolver


def build_decision_tree_model(dataset_directory, model_file):
    dt = DecisionTreeSolver.DecisionTreeSolver()
    dt.train(dataset_directory, model_file)

def predict_with_decision_tree(dataset_directory, model_file):
    dt = DecisionTreeSolver.DecisionTreeSolver()
    dt.predict(dataset_directory, model_file)

def build_NaiveBayes_model(dataset_directory, model_file):
    nb = NaiveBayesSolver.NaiveBayesSolver()
    nb.train(dataset_directory, model_file)

def predict_with_NaiveBayes(dataset_directory, model_file):
    nb = NaiveBayesSolver.NaiveBayesSolver()
    nb.predict(dataset_directory, model_file)

if __name__ == "__main__":
    # (mode, technique, dataset_directory, model_file) = sys.argv[1:5]
    (mode, technique, dataset_directory, model_file) = ('test', 'bayes', 'test', 'output/model-bayes.json')

    if mode == 'train' and technique == 'bayes':
        build_NaiveBayes_model(dataset_directory, model_file)

    elif mode == 'test' and technique == 'bayes':
        predict_with_NaiveBayes(dataset_directory, model_file)

    elif mode == 'train' and technique == 'dt':
        build_decision_tree_model(dataset_directory, model_file)

    elif mode == 'test' and technique == 'dt':
        predict_with_decision_tree(dataset_directory, model_file)



    # test_dir = '/home/subhash/Courses/AI/Assignment4/spam-classification/part1/test'
    #
    # dirs = os.listdir(test_dir)
    # for class_dir_name in dirs:
    #     total_test_cases = 0
    #     correct_predictions = 0
    #     for f in os.listdir(os.path.join(test_dir, class_dir_name )):
    #         total_test_cases += 4
    #         predicted_class = bayes_classifier.predict_simple(os.path.join(os.path.join(test_dir, class_dir_name ),f))
    #         if predicted_class == class_dir_name:
    #             correct_predictions += 4
    #     print "Prediction Accuracy for class: %s " % class_dir_name
    #     print "### Total test cases: %d,  " % (total_test_cases)
    #     print "### Correct classification: %d,  " % (correct_predictions)
    #     print "### Accuracy: %.2f,  " % (float(correct_predictions)/total_test_cases)

    # bayes_classifier.predict_simple('/home/subhash/Courses/AI/Assignment4/spam-classification/part1/test/notspam/0002.b3120c4bcbf3101e661161ee7efcb8bf')

