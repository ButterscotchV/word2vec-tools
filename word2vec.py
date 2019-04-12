#import unidecode
import gensim
import logging
import os

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def show_file_contents(input_file):
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            print(line)
            break


def read_input(input_file):
    """This method reads the input file"""

    logging.info("Reading file \"{0}\", this may take a while...".format(input_file))
    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(line) <= 0:
                continue

            if i % 10000 == 0:
                logging.info("Read {0} lines...".format(i))

            # do some pre-processing and return list of words for each review
            # text
            #words = list(filter(None, unidecode.unidecode(line.lower().replace('\r', '').replace('\n', '')).split(' ')))
            words = list(filter(None, line.lower().strip().replace('\r', '').replace('\n', '').split(' ')))

            if len(words) <= 0:
                continue

            yield words


if __name__ == '__main__':
    abspath = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(abspath, "tags.txt")

    # read the tokenized reviews into a list
    # each review item becomes a serries of words
    # so this becomes a list of lists
    documents = list(read_input(data_file))
    logging.info("Done reading data file")

    #for doc in documents:
        #logging.info("Doc: " + str(doc))

    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        documents,
        size=128,
        window=128,
        min_count=1,
        workers=16)
    model.train(documents, total_examples=len(documents), epochs=15)

    if not os.path.exists(os.path.join(abspath, "vectors")):
        os.makedirs(os.path.join(abspath, "vectors"))

    # save the word vectors
    model.wv.save(os.path.join(abspath, "vectors/default_vectors"))

    # save the model
    model.save(os.path.join(abspath, "vectors/default_model"))
