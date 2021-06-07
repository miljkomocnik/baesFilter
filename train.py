from Model import Model
import configparser
import pickle


DEFAULT_FILENAME = "model"
DEFAULT_ALFA = 1


def make_vocab_of_unique_words(train):
    vocab = []
    for sentence in train:
        sentence_as_list = sentence.split()
        for word in sentence_as_list:
            vocab.append(word)
    vocab = list(dict.fromkeys(vocab))
    return vocab


def calculate_prob(vocab, train, len_of_whole_vocab, alfa):
    dict_prob = {}
    for w in vocab:
        emails_with_w = 0
        for sentence in train:
            if w in sentence:
                emails_with_w += 1

        total_ = len(train)
        rate = (emails_with_w + alfa) / (total_ + len_of_whole_vocab * alfa)  # smoothing applied
        dict_prob[w.lower()] = rate

    return dict_prob


def save_model(model: Model, filename):
    try:
        pickle.dump(model, open(filename, 'wb'))
        print("Model created successfully!")
    except ImportError:
        print("ImportError error while saving the model!")


def train_model():
    config = configparser.ConfigParser()
    if len(config.read('config.ini')) > 0:
        filename = config['model']['filename']
        alfa = int(config['model']['alfa'])
    else:
        filename = DEFAULT_FILENAME
        alfa = DEFAULT_ALFA
        config['model'] = {"filename": filename, "alfa": 1}
        with open('config.ini', 'w') as configfile:
            config.write(configfile)

    train_spam = ['send us your password', 'review our website', 'send your password', 'send us your account']
    train_ham = ['Your activity report', 'benefits physical activity', 'the importance vows']
    test_emails = {'spam': ['renew your password', 'renew your vows'],
                   'ham': ['benefits of our account', 'the importance of physical activity']}
    stop_key = ['us', 'the', 'of', 'your']

    vocab_spam_train = make_vocab_of_unique_words(train_spam)
    vocab_ham_train = make_vocab_of_unique_words(train_ham)

    # make a vocabulary of unique words that occur in known ham emails and calculate prob
    dict_spamicity = calculate_prob(vocab_spam_train, train_spam, len(vocab_spam_train)+len(vocab_ham_train), alfa)
    total_spam = len(train_spam)
    # make a vocabulary of unique words that occur in known ham emails and calculate prob
    dict_hamicity = calculate_prob(vocab_ham_train, train_ham, len(vocab_spam_train)+len(vocab_ham_train), alfa)
    total_ham = len(train_ham)

    # probability that a given email is a spam or a ham
    prob_spam = total_spam / (total_spam + total_ham)
    prob_ham = total_ham / (total_spam + total_ham)

    model = Model(dict_spamicity, dict_hamicity, prob_spam, prob_ham, total_spam, total_ham, stop_key, alfa)
    save_model(model, filename)


if __name__ == '__main__':
    train_model()
