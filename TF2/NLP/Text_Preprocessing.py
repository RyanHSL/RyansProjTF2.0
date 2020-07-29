from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Create a simple sentence list
sentences = [
    "Make America Great Again!",
    "Salute to America!",
    "Nobody knows more than I do.",
    "Fake News!"
]
#Define the maximum vocabulary size which will be 20000
MAX_VOCAB_SIZE = 20000 #Specify how many words we want our token either to actually keep
#Instantiate a Tokenizer object and pass in the max vocab size.
#Fit the sentences list
#Transform the text to sequences
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
#Print the sequences
print(sequences)
#Show the word index mapping
print(tokenizer.word_index)
#Pad sequence data and print it using default parameters
data = pad_sequences(sequences) #The maxlen is the length of the longest sentence by default
# print(data)
#Try Max_Sequence_Length and post padding
data = pad_sequences(sequences, maxlen=6, padding="post")
# print(data) #The padding is prepadding by default, because RNN remembers the last several terms
#Try extra paddings
data = pad_sequences(sequences, maxlen=7)
# print(data)
#Truncation
data = pad_sequences(sequences, maxlen=4)
# print(data)
#Truncation using post truncation
data = pad_sequences(sequences, maxlen=4, truncating="post")
print(data)
