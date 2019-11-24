import logging
import os

import gensim
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
from pandas import DataFrame

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

    logging.info("Creating TSNE graph...")
    tsne = TSNE(n_components=2, n_jobs=8)
    X_tsne = tsne.fit_transform(X)
    logging.info("Done creating TSNE graph!")

    logging.info("Creating DataFrame and graph...")
    df = DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    logging.info("Done creating DataFrame and graph!")

    logging.info("Plotting points...")
    ax.scatter(df['x'], df['y'])

    for word, pos in df.iterrows():
        ax.annotate(word, pos)
    logging.info("Done plotting points!")

    logging.info("Done generating graph! Saving and displaying the graph....")
    fig.savefig(os.path.join(abspath, "graph.png"))
    plt.show()
