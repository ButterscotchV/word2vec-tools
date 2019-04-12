import tensorflow as tf
import numpy as np
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

    embeddings_vectors = np.stack(list(X))
    logging.info("Done processing vocab!")

    logging.info("Processing with TensorFlow...")
    g_1 = tf.Graph()

    with g_1.as_default():
        # Create some variables.
        emb = tf.Variable(embeddings_vectors, name='word_embeddings')

        # Add an op to initialize the variable.
        init_op = tf.compat.v1.global_variables_initializer()

        # Add ops to save and restore all the variables.
        saver = tf.compat.v1.train.Saver()

        # Later, launch the model, initialize the variables and save the
        # variables to disk.
        with tf.compat.v1.Session(graph=g_1) as sess:
            sess.run(init_op)

        # Save the variables to disk.
            save_path = saver.save(sess, os.path.join(abspath, "tensorboard-dir/model.ckpt"))
            print("Model saved in path: %s" % save_path)

    words = '\n'.join(vocab)

    with open(os.path.join(abspath, "tensorboard-dir/metadata.tsv"), "w", encoding="utf-8") as f:
        f.write(words)
    logging.info("Done processing with TensorFlow!")
