import gensim
import logging
import os

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


if __name__ == '__main__':
    abspath = os.path.dirname(os.path.abspath(__file__))

    logging.info("Loading model...")
    model = gensim.models.Word2Vec.load(os.path.join(abspath, "vectors/default_model"))
    logging.info("Done loading model!")

    logging.info("Generating graph...")
    logging.info("Processing vocab...")
    vocab = list(model.wv.vocab)
    X = model[vocab]
    logging.info("Done processing vocab!")

    logging.info("Converting to TXT...")
    strings = list()

    for word in model.wv.vocab:
        string_list = list()
        
        for vector_val in model[word]:
            string_list.append(str(vector_val))
        
        strings.append(word + ' ' + ' '.join(string_list))

    word_vectors = '\n'.join(strings)

    with open(os.path.join(abspath, "wordvectors.txt"), "w", encoding="utf-8") as f:
        f.write(word_vectors)
    logging.info("Done converting to TXT!")
