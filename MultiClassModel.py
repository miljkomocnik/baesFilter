class MultiClassModel:

    def __init__(self, list_of_dict, list_of_prob, stop_words, list_of_totals, alfa, number_of_class, class_names):
        self.list_of_dict = list_of_dict
        self.list_of_prob = list_of_prob
        self.stop_words = stop_words
        self.list_of_totals = list_of_totals
        self.alfa = alfa
        self.number_of_class = number_of_class
        self.class_names = class_names
