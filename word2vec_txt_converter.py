import logging
import os

import gensim

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

    with open(os.path.join(abspath, "tag-vectors.txt"), "w+", encoding="utf-8") as f:
        for word in model.wv.vocab:
            vector_components = list()

            for vector_val in model[word]:
                vector_components.append(str(vector_val))

            f.write(word + ' ' + (' '.join(vector_components)) + '\n')

    logging.info("Done converting to TXT!")
