import math
import pickle
from Model import Model
import configparser
import train


# function to multiply all word probs together
def mult(list_: list):
    total_prob = 1
    for i in list_:
        total_prob = total_prob * math.log(i)
    return total_prob


def n_bayes(email: list, model: Model):
    prob_s = []
    prob_h = []
    pr_s = model.prob_spam
    pr_h = model.prob_ham
    prob_s.append(pr_s)
    prob_h.append(pr_h)

    for word in email:
        try:
            pr_ws = model.dict_spamicity[word]
        except KeyError:
            # Apply smoothing for word not seen in spam training data
            pr_ws = model.alfa / (
                    model.total_spam +
                    (len(model.dict_hamicity.keys()) + len(model.dict_spamicity.keys()))
                    * model.alfa)

        try:
            pr_wh = model.dict_hamicity[word]
        except KeyError:
            # Apply smoothing for word not seen in ham training data
            pr_wh = model.alfa / (
                    model.total_ham +
                    (len(model.dict_hamicity.keys()) + len(model.dict_spamicity.keys()))
                    * model.alfa)

        prob_s.append(pr_ws)
        prob_h.append(pr_wh)

    prob_x = mult(prob_s) + mult(prob_h)
    spam = mult(prob_s) / prob_x
    ham = mult(prob_h) / prob_x
    print(spam)
    print(ham)

    final_classification = spam > ham
    return final_classification


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
    loaded_model: Model = load_model()

    # remove new and stop words
    reduced_email = []
    for word in email_words:
        if word not in loaded_model.stop_words:
            reduced_email.append(word)
    print(reduced_email)
    print(
        n_bayes(reduced_email, loaded_model)
    )


if __name__ == '__main__':
    # define training data
    train.train_model()
    test_email("review your password")
