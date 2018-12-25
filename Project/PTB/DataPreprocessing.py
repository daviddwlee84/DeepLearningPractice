import codecs
import collections
from operator import itemgetter

def _getVariables(mode):
    if mode == "PTB":            # PTB data preprocessing
        RAW_DATA = "PTB_data/ptb.train.txt"  # Training data
        VOCAB_OUTPUT = "output_vocab/ptb.vocab"           # Output vocabulary
        VOCAB_SIZE = 0 # didn't use in this mode
    elif mode == "TRANSLATE_ZH": # Translation Chinese Corpus
        RAW_DATA = "TED_data/train.txt.zh"
        VOCAB_OUTPUT = "output_vocab/zh.vocab"
        VOCAB_SIZE = 4000
    elif mode == "TRANSLATE_EN": # Translation English Corpus
        RAW_DATA = "TED_data/train.txt.en"
        VOCAB_OUTPUT = "output_vocab/en.vocab"
        VOCAB_SIZE = 10000
    
    return RAW_DATA, VOCAB_OUTPUT, VOCAB_SIZE

def generateVocabulary(mode, raw_data, vocab_output, vocab_size):
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
    if mode == "PTB":
        # We'll need to add <eos> at the end of sentence (new line) later, so we add it to vocabulary first.
        sorted_words = ["<eos>"] + sorted_words
    elif mode in ["TRANSLATE_EN", "TRANSLATE_ZH"]:
        # We need <eos> and <unk> as the start of the sentence and <sos> as the low frequency stop words
        sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words
        if len(sorted_words) > vocab_size:
            sorted_words = sorted_words[:vocab_size]
    
    with codecs.open(vocab_output, 'w', 'utf-8') as file_output:
        for word in sorted_words:
            file_output.write(word + "\n")

def main():
    modes = ["PTB", "TRANSLATE_EN", "TRANSLATE_ZH"]
    for mode in modes:
        var = _getVariables(mode)
        generateVocabulary(mode, *var)

if __name__ == "__main__":
    main()