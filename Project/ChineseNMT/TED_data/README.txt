原始文件(train.raw.en, train.raw.zh)是由 https://wit3.fbk.eu/mt.php?release=2015-01 下載的XML文件中提取的文本。
英語訓練文件(train.txt.en)使用Moses tokenizer進行切詞: https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl
中文訓練文件(train.txt.zh)按字符切詞。
示例程序中只需要用到train.txt.en和train.txt.zh，原始文件僅供參考之用。