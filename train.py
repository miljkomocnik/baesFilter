from Model import Model
import configparser
import pickle


def make_vocab_of_unique_words_and_prob(train):
    vocab = []
    for sentence in train:
        sentence_as_list = sentence.split()
        for word in sentence_as_list:
            vocab.append(word)
    vocab = list(dict.fromkeys(vocab))
    dict_prob = {}
    for w in vocab:
        emails_with_w = 0
        for sentence in train:
            if w in sentence:
                emails_with_w += 1

        total_ = len(train)
        spamicity = (emails_with_w + 1) / (total_ + 2)  # smoothing applied
        dict_prob[w.lower()] = spamicity

    return dict_prob, total_


def save_model(model: Model):
    config = configparser.ConfigParser()
    if len(config.read('config.ini')) > 0:
        filename = config['model']['filename']
    else:
        filename = "model"
        config['model'] = {"filename": filename}
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
    try:
        pickle.dump(model, open(filename, 'wb'))
        print("Model created successfully!")
    except ImportError:
        print("ImportError error while saving the model!")


def train_model():
    train_spam = ['send us your password', 'review our website', 'send your password', 'send us your account']
    train_ham = ['Your activity report', 'benefits physical activity', 'the importance vows']
    test_emails = {'spam': ['renew your password', 'renew your vows'],
                   'ham': ['benefits of our account', 'the importance of physical activity']}
    stop_key = ['us', 'the', 'of', 'your']

    # make a vocabulary of unique words that occur in known ham emails and calculate prob
    dict_spamicity, total_spam = make_vocab_of_unique_words_and_prob(train_spam)
    # make a vocabulary of unique words that occur in known ham emails and calculate prob
    dict_hamicity, total_ham = make_vocab_of_unique_words_and_prob(train_ham)

    # probability that a given email is a spam or a ham
    prob_spam = len(train_spam) / (len(train_spam) + (len(train_ham)))
    prob_ham = len(train_ham) / (len(train_spam) + (len(train_ham)))

    model = Model(dict_spamicity, dict_hamicity, prob_spam, prob_ham, total_spam, total_ham, stop_key)
    save_model(model)


if __name__ == '__main__':
    train_model()
