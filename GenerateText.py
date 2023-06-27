import os
from PyQt6.QtCore import QThread, pyqtSignal
from gensim.models import Word2Vec

class GenerateText(QThread):
    increaseProgressBar = pyqtSignal(float)
    sendLogMessage = pyqtSignal(str, str)
    setGeneratedText = pyqtSignal(str)

    def __init__(self, query_entry, temperature, num_words, gpu_enabled):
        QThread.__init__(self)
        self.query_entry = query_entry
        self.temperature = temperature
        self.num_words = num_words
        self.gpu_enabled = gpu_enabled


    def run(self):
        # Reset the progress bar
        self.increaseProgressBar.emit(0)
        self.sendLogMessage.emit("", "")

        # --- Load the word2vec model ---
        if not os.path.exists(f'{os.getcwd()}/ProcessedData/word2vec-model'):
            self.sendLogMessage.emit("Model not found. Please train the model first.", "red")
            return
        else:
            # Load the word2vec model
            model = Word2Vec.load(f'{os.getcwd()}/ProcessedData/word2vec-model')
            self.sendLogMessage.emit("Model loaded.", "green")
            self.increaseProgressBar.emit(3)

        if self.query_entry == "":
            self.sendLogMessage.emit("Please enter a word.", "red")
            self.increaseProgressBar.emit(0)
            return

        try:
            # Save the original query entry
            original_query_entry = self.query_entry
            
            # Make the query lowercase
            self.query_entry = self.query_entry.lower()
            # Remove punctuation from the query
            self.query_entry = self.query_entry.replace(".", "").replace("'", "").replace("’", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(";", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("'", "").replace('"', "").replace("-", "").replace("_", "").replace("/", "").replace("/", "").replace("|", "").replace("<", "").replace(">", "").replace("=", "").replace("+", "").replace("*", "").replace("&", "").replace("^", "").replace("%", "").replace("$", "").replace("#", "").replace("@", "")

            result_str = ""
            query = self.query_entry.split()

            # Save the punctuation and capital letters
            punctuation_idx = []
            capital_letters_idx = []
            punctuation_char = []
            original_query_entry_array = original_query_entry.split()
            word_idx = 0
            for word in original_query_entry_array:
                for char in word.strip():
                    if char in [".", ",", "?", "!"]:
                        punctuation_idx.append(word_idx)
                        punctuation_char.append(char)
                    if char.isupper():
                        capital_letters_idx.append(word_idx)
                word_idx += 1

            # Predict the most similar word
            id = 0
            if self.temperature == 0:
                for word in query:
                    if not word in ["is", "to","are", "the", "a", "an", "and", "or", "but", "for", "nor", "so", "yet", "at", "around", "by", "after", "along", "for", "from", "of", "on", "to", "with", "without", "over", "under", "above", "below", "up", "down", "in", "out", "off", "on", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "don", "should", "now"]:
                        if word == "he": sim_word = "she"
                        elif word == "his": sim_word = "her"
                        else: sim_word = model.wv.most_similar(word, topn=1)[0][0]
                        id += 1
                        sim_word += " "
                        result_str += sim_word
                    else:
                        result_str += word + " "

                # Add the punctuation back
                result_str_array = result_str.split()
                word_idx = 0
                array_idx = 0
                current_char_idx = 0
                for word in result_str_array:
                    if len(punctuation_idx) >= array_idx and word_idx == punctuation_idx[array_idx]:
                        char_idx = current_char_idx + len(word)
                        result_str = result_str[:char_idx] + punctuation_char[array_idx] + result_str[char_idx:]
                        array_idx += 1
                        current_char_idx += 1
                    word_idx += 1
                    current_char_idx += len(word) + 1

                # Capitalize the words that were capitalized in the query
                result_str_array = result_str.split()
                word_idx = 0
                array_idx = 0
                current_char_idx = 0
                for word in result_str_array:
                    if len(capital_letters_idx) > array_idx and word_idx == capital_letters_idx[array_idx]:
                        char = result_str[current_char_idx].capitalize()
                        result_str = result_str[:current_char_idx] + char + result_str[current_char_idx+1:]
                        array_idx +=1
                    word_idx += 1
                    current_char_idx += len(word) + 1

            else:
                similar_words = model.wv.most_similar(query, topn=self.num_words)
                for tup in similar_words:
                    result_str += f"{tup[0]} - {round(100*tup[1], 1)}%\n"
            self.setGeneratedText.emit(str(result_str[:-1]))
            self.increaseProgressBar.emit(97)
            self.sendLogMessage.emit("Done!", "green")
        except KeyError as error:
            if 'Key' in str(error):
                if self.temperature == 0:
                    self.sendLogMessage.emit("The word '" + query[id] + "' is not in the vocabulary.", "red")
                else:
                    self.sendLogMessage.emit("The word(s) '" + self.query_entry + "' is not in the vocabulary.", "red")
                self.setGeneratedText.emit("")
                self.increaseProgressBar.emit(0)
                return
            else:
                raise error
