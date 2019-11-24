import logging
import os

import gensim

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def show_file_contents(input_file):
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            print(line)
            break


class LineReader:
    """Iterator that reads and processes a file."""

    def __init__(self, file):
        self.file = open(file, 'r', encoding='utf-8')

    def __iter__(self):
        return self

    def __next__(self):
        line = self.file.readline()
        if line is None or len(line) <= 0:
            raise StopIteration

        clean_line = line.strip().replace('\r', '').replace('\n', '')

        image_id, *words = list(filter(None, clean_line.split(' ')))
        return words

    def close(self):
        self.file.close()


if __name__ == '__main__':
    abspath = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(abspath, "image-tags.txt")

    # Count the lines that will be read
    with open(data_file, 'r', encoding='utf-8') as f:
        lineCount = sum(1 for line in f)

    # Create a line iterator
    lineIterator = LineReader(data_file)

    # Build vocabulary and train model
    model = gensim.models.Word2Vec(
        lineIterator,
        size=128,
        window=128,
        min_count=1,
        workers=8)
    model.train(lineIterator, total_examples=lineCount, epochs=15)

    # Close the line reader
    lineIterator.close()

    if not os.path.exists(os.path.join(abspath, "vectors")):
        os.makedirs(os.path.join(abspath, "vectors"))

    # save the word vectors
    model.wv.save(os.path.join(abspath, "vectors/default_vectors"))

    # save the model
    model.save(os.path.join(abspath, "vectors/default_model"))
