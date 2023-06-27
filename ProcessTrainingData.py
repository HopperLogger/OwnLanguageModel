import json
import os
import random
import re
from tkinter import messagebox
import docx
from PyQt6.QtCore import QThread, pyqtSignal
import enchant
import time
from gensim.models import Word2Vec
import numpy as np

class ProcessTrainingData(QThread):
    increaseProgressBar = pyqtSignal(float)
    sendLogMessage = pyqtSignal(str, str)
    updateStats = pyqtSignal(str)

    def __init__(self, training_folder, embedding_path, load_dump, num_documents_to_load, check_spelling, epochs):
        QThread.__init__(self)
        self.training_folder = training_folder
        self.embedding_path = embedding_path
        self.load_dump = load_dump
        self.check_spelling = check_spelling
        self.num_documents_to_load = num_documents_to_load
        self.epochs = epochs

    def saveDataChunks(self, arr: list, chunk_size: int, file_name: str, type: str) -> None:
        """
        Saves the data in chunks to the ProcessedData folder.

        Parameters:\n
            arr (list): The array to save.
            chunk_size (int): The size of each chunk.
            file_name (str): The name of the file to save.
            type (str): The type of the array. Either 'json' or 'numpy'.

        Returns:\n
            None
        """
        num_files = int(len(arr) // chunk_size) + 1
        if type == "json":
            for i in range(0, num_files):
                chunk = arr[chunk_size*i:chunk_size*(i+1)]
                with open(f'{os.getcwd()}/ProcessedData/{file_name}-{i}.json', 'w') as file:
                    json.dump(chunk, file)
        elif type == "numpy":
            for i in range(0, num_files):
                chunk = arr[chunk_size*i:chunk_size*(i+1)]
                training_data = {}
                for j in range(0, len(chunk)):
                    training_data[f'sentence{j}'] = chunk[j]
                np.savez_compressed(f'{os.getcwd()}/ProcessedData/{file_name}-{i}.npz', **training_data)
        else:
            raise Exception("Invalid type. Must be either 'json' or 'numpy'.")
            
    def loadDataChunks(self, file_name: str) -> list:
        """
        Loads data in chunks from the ProcessedData folder.

        Parameters:\n
            file_name (str): The name of the files to load.

        Returns:\n
            list: The loaded data.
        """
        num_files = len([name for name in os.listdir(f'{os.getcwd()}/ProcessedData') if name.startswith(file_name)])
        arr = []
        if os.path.exists(f'{os.getcwd()}/ProcessedData/{file_name}-0.json'):
            for i in range(0, num_files):
                with open(f'{os.getcwd()}/ProcessedData/{file_name}-{i}.json', 'r') as file:
                    chunk = json.load(file)
                    arr.extend(chunk)
        elif os.path.exists(f'{os.getcwd()}/ProcessedData/{file_name}-0.npz'):
            for i in range(0, num_files):
                chunk = np.load(f'{os.getcwd()}/ProcessedData/{file_name}-{i}.npz')
                arr += list(chunk.values())
        else:
            raise Exception("File does not exist.")
        
        return arr

    def loadTrainingData(self, training_folder: str, num_documents_to_load: int = 0, load_dump: bool = True, check_spelling: bool = False) -> list:
        """
        Loads the training data from the training folder and preprocesses it.

        Parameters:\n
            training_folder (str): The path to the training folder.
            num_documents_to_load (int): The number of documents to load from the training folder and dump.
            load_dump (bool): Whether to load the story dump or not.
            check_spelling (bool): Whether to check the spelling of the sentences and only keep correctly spelled, English sentences or not.
        
        Returns:\n
            list: A list of all the cleaned sentences from the training data.
        """
        self.sendLogMessage.emit("Loading the training data...", "yellow")
        
        # Loop through all files in the folder
        all_documents = []
        for text_document in os.listdir(training_folder):
            # Load the word documents
            if (text_document.endswith('.docx')):
                doc = docx.Document(os.path.join(training_folder, text_document))
                # Extract the text content
                text = "\n".join([para.text for para in doc.paragraphs])
                all_documents.append(text)
                self.sendLogMessage.emit("Loaded word document: " + text_document, "green")

            # Load the text documents
            elif (text_document.endswith('.txt')):
                with open(training_folder + "/" + text_document, 'r') as file:
                    doc = file.read()
                    all_documents.append(doc)
                    self.sendLogMessage.emit("Loaded text document: " + text_document, "green")

        # Load the story dump
        if load_dump == True and os.path.exists(training_folder + '/' + 'story-dump.json'):
            with open(training_folder + '/' + 'story-dump.json', 'r') as story_dump:
                all_stories, last_saved_story = json.load(story_dump)
                all_documents = all_documents + all_stories
            self.sendLogMessage.emit("Story dump loaded.", "green")

        # Decrease the size of the array
        if len(all_documents) > num_documents_to_load and num_documents_to_load != 0:
            all_documents = all_documents[:num_documents_to_load]
            self.sendLogMessage.emit(f"Reduced the number of documents to {num_documents_to_load}.", "yellow")

        self.sendLogMessage.emit(f"Training data loaded. Number of documents: {len(all_documents)}", "green")
        
        # --- Save the total number of loaded documents to save later ---
        num_total_documents = len(all_documents)

        # --- Remove special characters from the documents ---
        self.sendLogMessage.emit("Cleaning the documents...", "yellow")
        all_documents = [document.lower().replace('\n', ' ').replace('"' , '').replace('“', '').replace('”', '')
                            .replace('’', '').replace('‘', '').replace('—', '').replace(',', '').replace('…', '')
                            .replace('•', '').replace('·', '').replace('!', '.').replace('?', '.').replace(';', '.')
                            .replace(':', '.').replace('(', '').replace('[', '').replace(']', '').replace(')', '')
                            .replace('{', '').replace('}', '').replace('=', '').replace('+', '').replace('-', '')
                            .replace('_', '').replace('|', '').replace('/', '').replace('/', '').replace('<', '')
                            .replace('>', '').replace('`', '').replace('~', '').replace('@', '').replace('#', '')
                            .replace('$', '').replace('%', '').replace('^', '').replace('&', '').replace('*', '')
                            .replace('...', '')
                            for document in all_documents]
        all_documents = [''.join([i for i in document if not i.isdigit()]) for document in all_documents] # Remove numbers
        all_documents = [' '.join(document.split()) for document in all_documents] # Remove multiple spaces
        self.increaseProgressBar.emit(2)
        self.sendLogMessage.emit("Documents cleaned.", "green")

        # --- Tokenize the documents into sentences and words ---
        def sentenceTokenizer(documents: list) -> list:
            """
            Tokenizes documents into sentences and words.

            Parameters:\n
                documents (list): A list of documents to tokenize.

            Returns:\n
                list: A list of tokenized sentences, where each sentence is a list of words.
                -> [['This', 'is', 'a', 'sentence'], ['This', 'is', 'another', 'sentence']]
            """
            tokenized_documents = []
            num_documents = len(documents)
            pr = 5 / num_documents # Calculate the progress bar increase per document
            for document in documents:
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', document) # Split the document into sentences
                tokenized_sentences = []
                for sentence in sentences:
                    words = sentence.split() # Split the sentence into words
                    words = [word.strip('.,?!') for word in words] # Remove punctuation from words
                    words = [word for word in words if word] # Remove empty words
                    # Detect and separate words stuck together with a period
                    new_words = []
                    for word in words:
                        stuck_words = re.findall(r'\b\w+\.\w+\b', word)
                        if stuck_words:
                            new_words += stuck_words
                        else:
                            new_words.append(word)
                    if len(new_words) <= 1: # Skip sentences with only one word
                        continue
                    tokenized_sentences.append(new_words)
                tokenized_documents += tokenized_sentences
                self.increaseProgressBar.emit(pr)
            return tokenized_documents

        self.sendLogMessage.emit("Tokenizing the documents...", "yellow")
        training_sentences = sentenceTokenizer(all_documents)
        self.sendLogMessage.emit("Documents tokenized.", "green")

        # Seperate words stuck together with a period
        self.sendLogMessage.emit("Seperating words stuck together with a period.", "yellow")
        clean_training_sentences = []
        for sentence in training_sentences:
            new_sentence = []
            for word in sentence:
                if re.findall(r'\b\w+\.\w+\b', word) != []:
                    period_index = word.find('.')
                    word_one = word[:period_index]
                    word_two = word[period_index+1:]
                    new_sentence.append(word_one)
                    new_sentence.append(word_two)
                else:
                    new_sentence.append(word)
            clean_training_sentences.append(new_sentence)

        self.sendLogMessage.emit("Training sentences converted to arrays. Here's a random one:", "green")
        self.sendLogMessage.emit(f"{str(random.choice(clean_training_sentences))}", "blue")

        valid_training_sentences = []

        # --- Filter out misspelled and German words ---
        if check_spelling == True:
            self.sendLogMessage.emit("Filtering out misspelled and German words...", "yellow")
            dict = enchant.Dict("en_US")
            num_training_sentences = len(clean_training_sentences)
            num_filtered_sentences = 0
            filtered_words = []
            pr = 10 / num_training_sentences # Calculate the progress bar increase per sentence
            for sentence in clean_training_sentences:
                if all(dict.check(word) for word in sentence):
                    valid_training_sentences.append(sentence)
                else:
                    num_filtered_sentences += 1
                    filtered_words += [word for word in sentence if not dict.check(word)]
                self.increaseProgressBar.emit(pr)
            self.sendLogMessage.emit(f"{'{:,}'.format(num_filtered_sentences)} of {'{:,}'.format(num_training_sentences)} German and misspelled sentences filtered out.", "green")
        else:
            valid_training_sentences += clean_training_sentences
            self.increaseProgressBar.emit(10)

        # --- Remove empty sentences ---
        self.sendLogMessage.emit("Removing empty sentences...", "yellow")
        valid_training_sentences = [sentence for sentence in valid_training_sentences if sentence != []]
        self.sendLogMessage.emit("Empty sentences removed.", "green")

        # --- Count the number of words in the valid training sentences ---
        num_words = 0
        for sentence in valid_training_sentences:
            num_words += len(sentence)

        # --- Save the stats ---
        with open(f'{os.getcwd()}/ProcessedData/training-data-stats.json', 'w') as f:
            json.dump(num_total_documents, f)

        self.sendLogMessage.emit(f"Sucessfully saved {'{:,}'.format(num_words)} new valid words of training data! Here are a few:", "green")
        random_words = ""
        for i in range(20):
            sentence = random.choice(valid_training_sentences)
            word = random.choice(sentence)
            random_words += word + " - "
        random_words = random_words[:-2]
        self.sendLogMessage.emit(f"{random_words}", "blue")
        self.increaseProgressBar.emit(2)
        self.updateStats.emit("training-data")

        return valid_training_sentences

    def createWord2VecModel(self, train_data: list, epochs: int = 30) -> Word2Vec:
        """
        Creates a Word2Vec model from the given training data.

        Parameters:\n
            train_data (list): The training data to use to train the model.
            epochs (int): The number of epochs to train the model for.
            
        Returns:\n
            model (Word2Vec): The trained Word2Vec model.
        """
        # Create the word2vec model
        start_time = time.time()
        model = Word2Vec(sentences=train_data, vector_size=100, window=5, min_count=1, workers=8, epochs=1)
        elapsed_time = time.time() - start_time
        time_in_sec = int(elapsed_time)
        eta = round((((epochs - 1)*time_in_sec)/60),1)
        self.sendLogMessage.emit("Created a new Word2Vec model.", "green")
        self.sendLogMessage.emit(f"Step 1 --- Took {time_in_sec}s --- ETA {eta} min(s).", "blue")

        # --- Check the folder ---
        if not os.path.exists(f'{os.getcwd()}/ProcessedData'):
            os.makedirs(f'{os.getcwd()}/ProcessedData')

        trained_epochs = 1
        pr = 75 / (epochs-1)
        for epoch in range(epochs-1):
            start_time = time.time()
            model.train(train_data, total_examples=model.corpus_count, epochs=1)
            trained_epochs += 1
            try:
                model.save(f'{os.getcwd()}/ProcessedData/word2vec-model')
                with open(f'{os.getcwd()}/ProcessedData/word2vec-model-steps.json', 'w') as f:
                    json.dump(trained_epochs, f)
            except PermissionError as error:
                self.sendLogMessage.emit("Could not save the model. Trying again...", "yellow")
                time.sleep(1)
                model.save(f'{os.getcwd()}/ProcessedData/word2vec-model')
                with open(f'{os.getcwd()}/ProcessedData/word2vec-model-steps.json', 'w') as f:
                    json.dump(trained_epochs, f)
            self.increaseProgressBar.emit(pr)
            # Show how long the save took
            elapsed_time = time.time() - start_time
            time_in_sec = int(elapsed_time)
            eta = round((((epochs - epoch)*time_in_sec)/60),1)
            self.sendLogMessage.emit(f"Step {trained_epochs} --- Took {time_in_sec}s --- ETA {eta} min(s).", "blue")
        self.sendLogMessage.emit("Word2Vec model sucessfully trained.", "green")
        self.increaseProgressBar.emit(1)
        self.updateStats.emit("model")

        return model

    def getPosEncodings(self, model: Word2Vec, train_data: list, max_seq_len: int = 100) -> list:
        """
        Generates positional encodings for each word in the training data and adds them to the word embeddings.

        Parameters:\n
            model (Word2Vec): The Word2Vec model that provides the word embeddings for each word in the train_data.
            train_data (list): The training data that will be used to generate the positional encodings.
            max_seq_len (int): The maximum length of a sentence in the training data. Default is 100.

        Returns:\n
            pos_encoded_train_data (list): The training data with the positional encodings added to each word embedding.
        """
        self.sendLogMessage.emit("Generating positional encodings...", "blue")
        # Generate positional encodings for each position in the sequence
        embedding_size = model.vector_size
        pos_encodings = np.zeros((max_seq_len, embedding_size)) # Create a 100x100 matrix
        for pos in range(max_seq_len): # For each position in the sequence
            for j in range(embedding_size):
                if j % 2 == 0:
                    pos_encodings[pos, j] = np.sin(pos / (10000 ** (2*j / embedding_size)))
                else:
                    pos_encodings[pos, j] = np.cos(pos / (10000 ** ((2*j - 1) / embedding_size)))

        # Add the positional encodings to each sentence array
        pos_encoded_train_data = []
        num_sentences = len(train_data)
        for sentence in train_data:
            print(f"Generating positional encodings for sentence {train_data.index(sentence)+1}/{num_sentences}")
            sentence_embeddings = []
            for i, word in enumerate(sentence):
                # Get the word vector before positional encoding
                word_vec = model.wv.get_vector(word)
                # Cap the length of the sentence to max_seq_len
                if i >= max_seq_len:
                    break
                # Add the positional encoding vector to the word vector
                word_vec_pos_encoded = word_vec + pos_encodings[i, :]
                # Add the pos-encoded word vector to the sentence_embeddings list
                sentence_embeddings.append(word_vec_pos_encoded)
            # Truncate the sentence to max_seq_len words
            sentence_length = min(len(sentence_embeddings), max_seq_len)
            # Add the pos-encoded sentence to the pos_encoded_train_data list
            pos_encoded_sentence = np.array(sentence_embeddings[:sentence_length])
            pos_encoded_train_data.append(pos_encoded_sentence)

        return pos_encoded_train_data
    
    def computeSelfAttention(self, train_data_vecs: list) -> list:
        """
        Computes the self-attention for each word in the training data.

        Parameters:\n
            train_data_vecs (list): The training data with the positional encodings added to each word embedding.
            
        Returns:\n
            new_train_data_vecs (list): The training data with the self-attention added to each word embedding.
        """
        new_train_data_vecs = []
        for sentence in train_data_vecs:
            new_word_vecs = []
            for word_vec in sentence:
                weights = [word_vec.dot(vec) for vec in sentence]
                normalized_weights = [(weight - min(weights)) / (max(weights) - min(weights)) for weight in weights]
                new_word = sum(weight * vec for weight, vec in zip(normalized_weights, sentence))
                new_word_vecs.append(new_word)
            new_train_data_vecs.append(new_word_vecs)
        return new_train_data_vecs

    def run(self) -> None:
        """
        Main function of the thread.
        Prepares the training data, 
        trains the Word2Vec model, 
        generates the positional encodings 
        and computes the self-attention.

        Parameters:\n
            None

        Returns:\n
            None
        """
        # --- Check the start situation ---

        # Check if the training folder is selected
        if self.training_folder == '' or self.training_folder == 'No training folder selected':
            self.sendLogMessage.emit("No training folder selected. Please select a training folder.", "red")
            return

        # Check if the ProcessedData folder exists
        if not os.path.exists(f'{os.getcwd()}/ProcessedData'):
            os.makedirs(f'{os.getcwd()}/ProcessedData')

        # Check if previous training data exists
        if os.path.exists(f'{os.getcwd()}/ProcessedData/training-data-0') and not messagebox.askyesno("Confirmation", "Previous Training Data found. Do you want to overwrite it?"):
            self.sendLogMessage.emit("Processing aborted.", "red")
            return

        # Check if the training folder contains documents
        num_documents = 0
        for document in os.listdir(self.training_folder):
            if (document.endswith('.docx') or document.endswith('.txt')):
                num_documents += 1
        # Documents don't exist, but story dump does
        if num_documents == 0 and self.load_dump == True and os.path.exists(self.training_folder + '/' + 'story-dump.json'):
            self.sendLogMessage.emit("Training folder contains no documents, continuing with just the story dump.", "yellow")
        # Documents don't exist, and story dump doesn't exist
        elif num_documents == 0 and self.load_dump == True:
            self.sendLogMessage.emit("Training folder contains no documents and no story dump. Please select a folder with documents.", "red")
            return
        # Documents don't exist, and no story dump but story dump is not selected
        elif num_documents == 0:
            self.sendLogMessage.emit("Training folder contains no documents. Please select a folder with documents.", "red")
            return
        # Documents exist, but no story dump
        elif self.load_dump == True and not os.path.exists(self.training_folder + '/' + 'story-dump.json'):
            answer = messagebox.askyesno("Confirmation", "No story dump found, but selected. Do you want to continue without the story dump?", icon='warning')
            if answer == False:
                self.sendLogMessage.emit("Processing aborted.", "red")
                return

        # --- Main functions ---
        if True:
            # Preprocess the training data
            train_data = self.loadTrainingData(self.training_folder, self.num_documents_to_load, self.load_dump, self.check_spelling)
            self.saveDataChunks(train_data, chunk_size=100000, type="json", file_name="training-data")
            self.sendLogMessage.emit("Saved training data.", "green")

            # Train the word2vec model
            word2vec_model = self.createWord2VecModel(train_data, self.epochs)
            self.sendLogMessage.emit("Saved Word2Vec model.", "green")

            # Pos encode the training data
            pos_encoded_train_data_vecs = self.getPosEncodings(word2vec_model, train_data, max_seq_len=100)
            self.saveDataChunks(pos_encoded_train_data_vecs, chunk_size=100000, type="numpy", file_name="pos-encoded-training-data-vecs")
            self.sendLogMessage.emit(f"Saved pos encoded training data vectors.", "green")

            # Compute the self attention
            attention_train_data_vecs = self.computeSelfAttention(pos_encoded_train_data_vecs)
            self.saveDataChunks(attention_train_data_vecs, chunk_size=100000, type="numpy", file_name="attention-training-data-vecs")
            self.sendLogMessage.emit("Saved attention data.", "green")
            self.increaseProgressBar.emit(5)