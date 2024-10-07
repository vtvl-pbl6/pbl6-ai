parent_dir="./hate_speech_text_span_detection"

rm -rf "$parent_dir/vncorenlp"
mkdir -p "$parent_dir/vncorenlp/models/wordsegmenter"

# Download VnCoreNLP
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr

# Move to the correct directory
mv VnCoreNLP-1.1.1.jar "$parent_dir/vncorenlp/"
mv vi-vocab "$parent_dir/vncorenlp/models/wordsegmenter"
mv wordsegmenter.rdr "$parent_dir/vncorenlp/models/wordsegmenter"
