import math
import pickle
import configparser
import train
from MultiClassModel import MultiClassModel


# function to multiply all word probs together
def mult(list_: list):
    total_prob = 1
    for i in list_:
        total_prob = total_prob * math.log(i)
    return total_prob


def multi_class_bayes(words: list, model: MultiClassModel):
    prob_x = 0
    classes = []
    for i in range(0, model.number_of_class):
        prob_c = [model.list_of_prob[i]]
        for word in words:
            try:
                pr_ws = model.list_of_dict[i][word]
            except KeyError:
                number_of_features = 0
                for vocab in model.list_of_dict:
                    number_of_features += len(vocab.keys())
                pr_ws = model.alfa / (
                        model.list_of_totals[i] +
                        number_of_features
                        * model.alfa)
            prob_c.append(pr_ws)
        mult_c = mult(prob_c)
        classes.append(mult_c)
        prob_x += mult_c
    results = []
    for c in classes:
        results.append(c/prob_x)
    return results


def load_model():
    config = configparser.ConfigParser()
    if len(config.read('config.ini')) > 0:
        filename = config['model']['filename']
        try:
            return pickle.load(open(filename, 'rb'))
        except ImportError:
            print("No model with filename=", filename, " present")
    else:
        print("No config.ini present")


def test_email(email: str):
    # split emails into distinct words
    email_words = list(dict.fromkeys(email.split()))
    loaded_model: MultiClassModel = load_model()

    # remove new and stop words
    reduced_email = []
    for word in email_words:
        if word not in loaded_model.stop_words:
            reduced_email.append(word)
    print(loaded_model.class_names)
    print(
        multi_class_bayes(reduced_email, loaded_model)
    )


if __name__ == '__main__':
    # define training data
    train.train_model()
    test_email("review your password")
