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

    def loadTrainingData(self, training_folder: str, num_documents_to_load: int = 0, load_dump: bool = True, check_spelling: bool = False) -> list:
        """
        Loads the training data from the training folder and preprocesses it.

        Parameters:\n
            training_folder (str): The path to the training folder.
            num_documents_to_load (int): The number of documents to load from the training folder and dump.
            load_dump (bool): Whether to load the story dump or not.
            check_spelling (bool): Whether to check the spelling of the sentences and only keep correctly spelled, English sentences or not.
        
        Returns and Saves:\n
            list: A list of all the cleaned sentences from the training data.
            -> [["Hello","world"],["This","is","the","second","sentence"]]
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
            pr = 15 / num_documents # Calculate the progress bar increase per document
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

        self.sendLogMessage.emit(f"Sucessfully saved {'{:,}'.format(num_words)} valid words of training data! Here are a few:", "green")
        random_words = ""
        for i in range(20):
            sentence = random.choice(valid_training_sentences)
            word = random.choice(sentence)
            random_words += word + " - "
        random_words = random_words[:-2]
        self.sendLogMessage.emit(f"{random_words}", "blue")
        self.increaseProgressBar.emit(3)
        self.updateStats.emit("training-data")

        # Save the valid training sentences
        with open(f'{os.getcwd()}/ProcessedData/clean-training-data.json', 'w') as file:
            json.dump(valid_training_sentences, file)
        self.sendLogMessage.emit("Saved the training data.", "green")
            
        return valid_training_sentences

    def createWord2VecModel(self, train_data: list, epochs: int = 30) -> Word2Vec:
        """
        Creates a Word2Vec model from the given training data.

        Parameters:\n
            train_data (list): The training data to use to train the model.
            epochs (int): The number of epochs to train the model for.
            
        Returns and Saves:\n
            model (Word2Vec): The trained Word2Vec model.
        """
        # Create the word2vec model
        start_time = time.time()
        self.sendLogMessage.emit("Creating a new Word2Vec model...", "yellow")
        model = Word2Vec(sentences=train_data, vector_size=100, window=5, min_count=1, workers=8, epochs=1)
        elapsed_time = time.time() - start_time
        time_in_sec = int(elapsed_time)
        eta = round((((epochs - 1)*time_in_sec)/60),1)
        self.sendLogMessage.emit(f"Step 1 --- Took {time_in_sec}s --- ETA: {eta} min(s).", "blue")

        # --- Check the folder ---
        if not os.path.exists(f'{os.getcwd()}/ProcessedData'):
            os.makedirs(f'{os.getcwd()}/ProcessedData')

        trained_epochs = 1
        pr = 30 / (epochs-1)
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
            self.sendLogMessage.emit(f"Step {trained_epochs} --- Took {time_in_sec}s --- ETA: {eta} min(s).", "blue")
        self.sendLogMessage.emit("Word2Vec model sucessfully trained and saved.", "green")
        self.updateStats.emit("model")

        return model

    def getPosEncodings(self, train_data: list, model: Word2Vec, max_seq_len: int = 100, start_idx: int = 0) -> list:
        """
        Generates positional encodings for each word in the training data and adds them to the word embeddings.

        Parameters:\n
            train_data (list): The training data that will be used to generate the positional encodings.
            model (Word2Vec): The Word2Vec model that provides the word embeddings for each word in the train_data.
            max_seq_len (int): The maximum length of a sentence in the training data. Default is 100.
            start_idx (int): The index to start at in the training data. Default is 0.

        Returns:\n
            None
            
        Saves:\n
            pos_encoded_train_data (list): The training data with the positional encodings added to each word embedding (in chunks).
            -> [[np.array,np.array,np.array],[np.array,np.array]]
        """
        self.sendLogMessage.emit("Generating positional encodings...", "yellow")
        
        embedding_size = model.vector_size
        
        # Generate positional encodings for each position in the sequence
        pos_encodings = np.zeros((max_seq_len, embedding_size))
        positions = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embedding_size, 2) * -(np.log(10000.0) / embedding_size))

        pos_encodings[:, 0::2] = np.sin(positions * div_term)
        pos_encodings[:, 1::2] = np.cos(positions * div_term)

        # Add the positional encodings to each sentence array
        batch = start_idx
        pos_encoded_train_data = []
        num_sentences = len(train_data)
        train_data = train_data[start_idx*100000:]
        pr = 40 / (len(train_data)/1000)
        offset = 1+start_idx*100000
        for i, sentence in enumerate(train_data):
            current_sentence_idx = i + offset
            if (current_sentence_idx % 10000 == 0):
                self.increaseProgressBar.emit(pr)
                self.sendLogMessage.emit(f"Generating positional encodings for sentence {'{:,}'.format(current_sentence_idx)}/{'{:,}'.format(num_sentences)}", "blue")
            
            sentence_embeddings = []
            for i, word in enumerate(sentence):
                # Cap the length of the sentence to max_seq_len
                if i >= max_seq_len:
                    break
                # Get the word vector before positional encoding
                word_vec = model.wv.get_vector(word)
                # Add the positional encoding vector to the word vector
                word_vec_pos_encoded = word_vec + pos_encodings[i, :]
                # Add the pos-encoded word vector to the sentence_embeddings list
                sentence_embeddings.append(word_vec_pos_encoded)
            # Truncate the sentence to max_seq_len words
            sentence_length = min(len(sentence_embeddings), max_seq_len)
            # Add the pos-encoded sentence to the pos_encoded_train_data list
            pos_encoded_sentence = np.array(sentence_embeddings[:sentence_length])
            pos_encoded_train_data.append(pos_encoded_sentence)
            
            # Save the current chunk of pos-encoded training data
            if len(pos_encoded_train_data) % 100000 == 0:
                self.sendLogMessage.emit(f"Saving batch {batch+1}/{num_sentences//100000+1} of pos encoded training data vectors...", "yellow")
                training_data = {}
                for i in range(100000):
                    training_data[f'sentence{i}'] = pos_encoded_train_data[i]
                np.savez(f'{os.getcwd()}/ProcessedData/pos-encoded-training-data-{batch}.npz', **training_data)
                pos_encoded_train_data = []
                batch += 1

        # Save the last chunk of pos-encoded training data
        self.sendLogMessage.emit(f"Saving batch {batch+1}/{num_sentences//100000+1} of pos encoded training data vectors...", "yellow")
        training_data = {}
        for i in range(len(pos_encoded_train_data)):
            training_data[f'sentence{i}'] = pos_encoded_train_data[i]
        np.savez(f'{os.getcwd()}/ProcessedData/pos-encoded-training-data-{batch}.npz', **training_data)
        self.sendLogMessage.emit(f"Sucessfully saved all the pos encoded training data vectors.", "green")
        self.increaseProgressBar.emit(5)

    def run(self) -> None:
        """
        Main function of the thread.
        Cleans the training data, 
        trains the Word2Vec model, 
        and generates the positional encodings.

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

        # Check if previous training data exists
        if os.path.exists(f'{os.getcwd()}/ProcessedData/clean-training-data.json') and not os.path.exists(f'{os.getcwd()}/ProcessedData/todo.json'):
            overwrite = messagebox.askyesno("Confirmation", "Previous Training Data found. Do you want to overwrite it?")
            if overwrite == True:
                # Delete all the files in the ProcessedData folder
                for file in os.listdir(f'{os.getcwd()}/ProcessedData'):
                    os.remove(f'{os.getcwd()}/ProcessedData/{file}')
                    print(f"Deleted {file}")
            else:
                self.sendLogMessage.emit("Processing aborted.", "red")
                return

        # Check if the ProcessedData folder exists
        if not os.path.exists(f'{os.getcwd()}/ProcessedData'):
            os.makedirs(f'{os.getcwd()}/ProcessedData')

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

        # --- TODO logic ---
        # Check if the TODO list exists
        if os.path.exists(f'{os.getcwd()}/ProcessedData/todo.json') and not messagebox.askyesno("Confirmation", "The previous processing was aborted, do you want to continue it?"):
            # Delete the TODO list
            os.remove(f'{os.getcwd()}/ProcessedData/todo.json')
        
        # Create the TODO list
        if not os.path.exists(f'{os.getcwd()}/ProcessedData/todo.json'):
            with open(f'{os.getcwd()}/ProcessedData/todo.json', 'w') as file:
                json.dump([1,1,1], file)
                    
        # Open the TODO list
        with open(f'{os.getcwd()}/ProcessedData/todo.json', 'r') as file:
            todo_list = json.load(file)
        train_data = None
        word2vec_model = None
            
        # --- Main functions ---
        # --- Preprocess the training data ---
        if todo_list[0]:
            train_data = self.loadTrainingData(self.training_folder, self.num_documents_to_load, self.load_dump, self.check_spelling)
            # Save the current progress in the TODO list
            with open(f'{os.getcwd()}/ProcessedData/todo.json', 'w') as file:
                json.dump([0,1,1], file)

        # --- Train the word2vec model ---
        if todo_list[1]:
            # Load the training data
            if train_data == None:
                with open(f'{os.getcwd()}/ProcessedData/clean-training-data.json', 'r') as file:
                    train_data = json.load(file)
                self.increaseProgressBar.emit(30)
            word2vec_model = self.createWord2VecModel(train_data, self.epochs)
            # Save the current progress in the TODO list
            with open(f'{os.getcwd()}/ProcessedData/todo.json', 'w') as file:
                json.dump([0,0,1], file)

        # --- Pos encode the training data ---
        if todo_list[2]:

            # Load the training data and word2vec model
            if train_data == None:
                with open(f'{os.getcwd()}/ProcessedData/clean-training-data.json', 'r') as file:
                    train_data = json.load(file)
                self.increaseProgressBar.emit(30)
                
            if word2vec_model == None:
                word2vec_model = Word2Vec.load(f'{os.getcwd()}/ProcessedData/word2vec-model')
                self.increaseProgressBar.emit(30)
                
            # Check if there are already pos encoded training data chunks
            start_idx = 0
            files = os.listdir(f'{os.getcwd()}/ProcessedData')
            for file in files:
                if file.startswith('pos-encoded-training-data'):
                    start_idx += 1
                        
            self.getPosEncodings(train_data, word2vec_model, max_seq_len=100, start_idx=start_idx)
            
            # Delete the TODO list
            os.remove(f'{os.getcwd()}/ProcessedData/todo.json')
            
        self.sendLogMessage.emit("Done!", "green")
