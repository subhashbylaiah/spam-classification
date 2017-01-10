**Spam Classification**

The objective of this project is to build a Spam classifier using 
classification algorithms such as Decision Trees(ID3 algorithm) and Naive Bayes.
We will build these algorithms from scratch without using any libraries.

The document model we will use here is a bag-of-words model. 
We are using two kinds of model 
1) based on word presence (whether a word appears in the document or not, 
this would make the input features binary)
2) based on word frequencies (frenquency of word occurence in the document, 
this would make the input features as continuous)


**Bag of Words Model:** 
A bag of words is a representation of a text(document here) as a "bag" of words 
without any consideration of its grammatical structure or the word order. 
It is simply a histogram over the words of the language, and each document is 
represented as a vector over these words and the entries in this vector simply 
corresponds to the presence or the absence of the corresponding word (when using
model 1 above or the frequency of the word occurence when using model 2 above) 
 
For the purposes of our program we will build these feature vectors based 
on the langauge model which only includes the words in the training corpus. 
 
**About the code**

The spam.py module has the main program that accepts arguments for the mode (train or test),
algorithm (Naive Bayes/Decision tree), path of files (train/test data)

The program can be run as below:


`./spam mode technique dataset-directory model-file
`

where 
* mode: test or train
* technique: bayes or dt
* dataset-directory: path of training corpus or the test dataset
* path where the model file is generated and saved (when training) or read from 
(when testing)

invokes the corresponding algorithm

This invokes the corresponding algorithm implementations

DecisionTreeSolver.py implements the ID3 decision tree algorithm

NaiveBayesSolver.py implements the Naive Bayes algorithm


**A brief Report**

Confusion Matrix for Decision Tree:
(Binary features model)

|---------------|--------------|---------|--------------------------------------|
|               |              |   Spam  | NotSpam   | Predicted                |
|   Actual      | Spam         |   1184  | 1                                    |
|               | Notspam      |   103   | 1266                                 |

Accuracy:

Prediction Accuracy for class spam: 1.00
Prediction Accuracy for class notspam: 0.92

Confusion Matrix for Decision Tree:
(Frequency based Features Model)

|---------------|--------------|---------|--------------------------------------|
|               |              |   Spam  | NotSpam   | Predicted                |
|   Actual      | Spam         |   1169  | 16                                   |
|               | Notspam      |   92   | 1277                                  |

Accuracy:

Prediction Accuracy for class spam: 0.99
Prediction Accuracy for class notspam: 0.93

Confusion Matrix for Naive Bayes:
(Binary Features Model)

|---------------|--------------|---------|--------------------------------------|
|               |              |   Spam  | NotSpam   | Predicted                |
|   Actual      | Spam         |   1139  | 46                                   |
|               | Notspam      |   16    | 1353                                 |

Accuracy:

Prediction Accuracy for class spam: 0.96
Prediction Accuracy for class notspam: 0.99


Confusion Matrix for Naive Bayes:
(Frequency based Features Model)

|---------------|--------------|---------|--------------------------------------|
|               |              |   Spam  | NotSpam   | Predicted                |
|   Actual      | Spam         |   1134  | 51                                   |
|               | Notspam      |   21    | 1358                                 |

Accuracy:

Prediction Accuracy for class spam: 0.96
Prediction Accuracy for class notspam: 0.98

