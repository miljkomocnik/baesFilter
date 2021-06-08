## References
* Datasets: https://www.baeldung.com/cs/spam-filter-training-sets
* Enron: http://www2.aueb.gr/users/ion/data/enron-spam/
* Laplace smoothing: https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece
* Understanding Naive Bayes algorithm: https://towardsdatascience.com/understanding-na%C3%AFve-bayes-algorithm-f9816f6f74c0
* Using log-probabilities for Naive Bayes: http://www.cs.rhodes.edu/~kirlinp/courses/ai/f18/projects/proj3/naive-bayes-log-probs.pdf

## Setup
1. pip install requirements.txt
2. download wordnet and punkt from http://www.nltk.org/nltk_data/
3. extract wordnet in C:\nltk_data\corpora
4. extract punkt in C:\nltk_data\tokenizers
5. create folder data, download and extract enron dataset in it

## To do
* add log-probabilities
* spilt data in training and testing
* calculate confusion matrix for tested data
* add custom lemmatisation algorithm
* create custom dataset
* test Laplace smoothing
* use different n-grams when creating models
* test different bayes methods
  * count word for every occurrence vs count as one occurrence
  * Additional bayes
    * Gaussian Naïve Bayes
    * Multinomial Naïve Bayes
    * Bernoulli Naïve Bayes
* optimisation for training methods
* optimisation for testing methods