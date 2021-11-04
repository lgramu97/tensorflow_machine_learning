

def run_bag_of_words():
    text = "this is a test to see if this test will work is is test a a"
    bag, vocab = bag_of_words(text,word_encoding=1)
    print(bag,'\n')
    print(vocab,'\n')

    positive_review = "I thought the movie was going to be bad but it was actually amazing"
    negative_review = "I thought the movie was going to be amazing but it was actually bad"

    pos_bag = bag_of_words(positive_review,1)
    neg_bag = bag_of_words(negative_review,1)

    print("Positive:", pos_bag,'\n')
    print("Negative:", neg_bag,'\n')


def bag_of_words(text,word_encoding):
    vocab = {}  # maps word to integer representing it
    words = text.lower().split(" ")  # create a list of all of the words in the text, well assume there is no grammar in our text for this example
    bag = {}  # stores all of the encodings and their frequency

    for word in words:
        if word in vocab:
            encoding = vocab[word]  # get encoding from vocab
        else:
            vocab[word] = word_encoding
            encoding = word_encoding
            word_encoding += 1
        
        if encoding in bag:
            bag[encoding] += 1
        else:
            bag[encoding] = 1
    
    return bag, vocab


def run():
    pass


if __name__ == '__main__':
    run_bag_of_words()