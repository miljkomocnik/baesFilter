import pickle
from Model import Model
import configparser
import train


# function to multiply all word probs together
def mult(list_: list):
    total_prob = 1
    for i in list_:
        total_prob = total_prob * i
    return total_prob


def n_bayes(email: list, model: Model):
    probs = []
    for word in email:
        pr_s = model.prob_spam
        try:
            pr_ws = model.dict_spamicity[word]
        except KeyError:
            # Apply smoothing for word not seen in spam training data, but seen in ham training
            pr_ws = 1 / (model.total_spam + 2)  # umesto dva ide broj reci

        pr_h = model.prob_ham
        try:
            pr_wh = model.dict_hamicity[word]
        except KeyError:
            # Apply smoothing for word not seen in ham training data, but seen in spam training
            pr_wh = (1 / (model.total_ham + 2))

        prob_word_is_spam_bayes = (pr_ws * pr_s) / ((pr_ws * pr_s) + (pr_wh * pr_h))
        probs.append(prob_word_is_spam_bayes)
    final_classification = mult(probs)
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
    email_words = email.split()
    loaded_model: Model = load_model()

    # remove new and stop words
    reduced_email = []
    for word in email_words:
        if word not in loaded_model.stop_words and \
                (word in loaded_model.dict_spamicity.keys() or
                 word in loaded_model.dict_hamicity.keys()):
            reduced_email.append(word)
    print(reduced_email)
    print(
        n_bayes(reduced_email, loaded_model)
    )


if __name__ == '__main__':
    # define training data
    train.train_model()
    test_email("renew renew your password")
