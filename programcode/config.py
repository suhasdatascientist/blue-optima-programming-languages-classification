import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "../data/code")

train_data_dir = os.path.join(data_dir, "train")
test_data_dir = os.path.join(data_dir, "test")
vocab_location = os.path.join(current_dir, "../data/vocab.txt")
vocab_tokenizer_location = os.path.join(current_dir, "../data/vocab_tokenizer")
word2vec_location = os.path.join(current_dir, "../data/word2vec.txt")
model_file_location = os.path.join(current_dir, "../data/models/model.json")
weights_file_location = os.path.join(current_dir, "../data/models/model.h5")

input_length = 500
word2vec_dimension = 100

