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

    logging.info("Converting to TSVs...")
    vector_strings = list()

    for vector in X:
        string_list = list()
        
        for vector_val in vector:
            string_list.append(str(vector_val))
        
        vector_strings.append('\t'.join(string_list))

    vectors = '\n'.join(vector_strings)

    if not os.path.exists(os.path.join(abspath, "tsv-dir")):
        os.makedirs(os.path.join(abspath, "tsv-dir"))

    with open(os.path.join(abspath, "tsv-dir/vectors.tsv"), "w", encoding="utf-8") as f:
        f.write(vectors)

    words = '\n'.join(vocab)

    with open(os.path.join(abspath, "tsv-dir/metadata.tsv"), "w", encoding="utf-8") as f:
        f.write(words)
    logging.info("Done converting to TSVs!")
