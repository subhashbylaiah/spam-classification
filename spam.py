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
    (mode, technique, dataset_directory, model_file) = sys.argv[1:5]
    # (mode, technique, dataset_directory, model_file) = ('test', 'bayes', 'test', 'output/model-bayes.json')

    if mode == 'train' and technique == 'bayes':
        build_NaiveBayes_model(dataset_directory, model_file)

    elif mode == 'test' and technique == 'bayes':
        predict_with_NaiveBayes(dataset_directory, model_file)

    elif mode == 'train' and technique == 'dt':
        build_decision_tree_model(dataset_directory, model_file)

    elif mode == 'test' and technique == 'dt':
        predict_with_decision_tree(dataset_directory, model_file)
