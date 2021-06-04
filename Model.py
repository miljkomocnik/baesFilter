class Model:

    def __init__(self, dict_spamicity, dict_hamicity, prob_spam, prob_ham, total_spam, total_ham, stop_words):
        self.dict_spamicity = dict_spamicity
        self.dict_hamicity = dict_hamicity
        self.prob_spam = prob_spam
        self.prob_ham = prob_ham
        self.total_spam = total_spam
        self.total_ham = total_ham
        self.stop_words = stop_words
