import codecs
import collections
from operator import itemgetter

# === Generate Vocabulary === #

RAW_DATA = "PTB_data/ptb.train.txt"     # Training data
VOCAB_OUTPUT = "output_vocab/ptb.vocab" # Output vocabulary

def generateVocabulary(raw_data, vocab_output):
    counter = collections.Counter()
    with codecs.open(raw_data, "r", "utf-8") as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1

    # Sorting by the frequency of words
    sorted_word_to_cnt = sorted(
        counter.items(), key=itemgetter(1), reverse=True)
    sorted_words = [x[0] for x in sorted_word_to_cnt]

    # Insert special symbol
    # We'll need to add <eos> at the end of sentence (new line) later, so we add it to vocabulary first.
    sorted_words = ["<eos>"] + sorted_words
    
    with codecs.open(vocab_output, 'w', 'utf-8') as file_output:
        for word in sorted_words:
            file_output.write(word + "\n")

# === Generate Training Data === #

def _getVariables2(mode):
    if mode == "PTB_TRAIN":        # PTB Training Data
        RAW_DATA = "PTB_data/ptb.train.txt"
        VOCAB = "output_vocab/ptb.vocab"
        OUTPUT_DATA = "train_data/ptb.train" # Output file which replaced words with their encode
    elif mode == "PTB_VALID":      # PTB Valication Data
        RAW_DATA = "PTB_data/ptb.valid.txt"
        VOCAB = "output_vocab/ptb.vocab"
        OUTPUT_DATA = "train_data/ptb.valid"
    elif mode == "PTB_TEST":       # PTB Testing Data
        RAW_DATA = "PTB_data/ptb.test.txt"
        VOCAB = "output_vocab/ptb.vocab"
        OUTPUT_DATA = "train_data/ptb.test"
    elif mode == "TRANSLATE_ZH":   # Chinese Translation Data
        RAW_DATA = "TED_data/train.txt.zh"
        VOCAB = "output_vocab/zh.vocab"
        OUTPUT_DATA = "train_data/train.zh"
    elif mode == "TRANSLATE_EN":   # English Translation Data
        RAW_DATA = "TED_data/train.txt.en"
        VOCAB = "output_vocab/en.vocab"
        OUTPUT_DATA = "train_data/train.en"
    
    return RAW_DATA, VOCAB, OUTPUT_DATA

def generateTrainingData(raw_data, vocab_input, output_data):
    # Load vocabulary, Construct encode projection of words
    with codecs.open(vocab_input, "r", "utf-8") as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]
    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    # If words not in vocabulary (i.e. low frequency words), replace it with <unk>
    def get_id(word):
        return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]

    fin = codecs.open(raw_data, "r", "utf-8")
    fout = codecs.open(output_data, 'w', 'utf-8')
    for line in fin:
        words = line.strip().split() + ["<eos>"]  # Read lines and add <eos>
        # Replace each word with its encoding
        out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
        fout.write(out_line)
    fin.close()
    fout.close()

def main():

    generateVocabulary(RAW_DATA, VOCAB_OUTPUT)

    modes = ["PTB_TRAIN", "PTB_VALID", "PTB_TEST"]
    for mode in modes:
        var = _getVariables2(mode)
        generateTrainingData(*var)

if __name__ == "__main__":
    main()