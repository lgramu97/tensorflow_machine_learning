def integer_encoding(text,word_encoding=1):
    '''
        This involves representing each word or character in a sentence as a unique integer 
        and maintaining the order of these words.
        This should hopefully fix the problem we saw before were we lost the order of words.
    '''
    vocab = {}  # maps word to integer representing it
    words = text.lower().split(" ") 
    encoding = []  

    for word in words:
        if word in vocab:
            code = vocab[word]  
            encoding.append(code) 
        else:
            vocab[word] = word_encoding
            encoding.append(word_encoding)
        word_encoding += 1
    
    return encoding, vocab


def run_integer_encoding():
    text = "this is a test to see if this test will work is is test a a"
    encoding, vocab = integer_encoding(text)
    print('Text: ', text)
    print('Encoding: ',encoding,'\n')
    print('Dictionary: ' ,vocab,'\n')

    positive_review = "I thought the movie was going to be bad but it was actually amazing"
    negative_review = "I thought the movie was going to be amazing but it was actually bad"

    pos_encode, _ = integer_encoding(positive_review)
    neg_encode, _  = integer_encoding(negative_review)

    print("Positive:", pos_encode,'\n')
    print("Negative:", neg_encode,'\n')


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


def run_bag_of_words():
    text = "this is a test to see if this test will work is is test a a"
    bag, vocab = bag_of_words(text,word_encoding=1)
    print('Text: ', text)
    print('Encoding: ',bag,'\n')
    print('Dictionary: ' ,vocab,'\n')

    positive_review = "I thought the movie was going to be bad but it was actually amazing"
    negative_review = "I thought the movie was going to be amazing but it was actually bad"

    pos_bag, _ = bag_of_words(positive_review,1)
    neg_bag, _ = bag_of_words(negative_review,1)

    print("Positive:", pos_bag,'\n')
    print("Negative:", neg_bag,'\n')


if __name__ == '__main__':
    print('BAG OF WORDS \n')
    run_bag_of_words()
    print('INTEGER ENCODING \n')
    run_integer_encoding()