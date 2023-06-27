# Own Language Model
This project is my first endeavor in developing a basic transformer-based language model. It incorporates a user-friendly GUI that empowers you to handpick your training documents. These documents undergo automated cleaning, tokenization, and subsequently contribute to the creation of a word embedding model, enriched with positional encodings. By specifying the desired training parameters, you can then train your very own PyTorch language model and leverage its capabilities to generate completions for input queries.

Please note that this project is currently a work in progress (WIP), with particular emphasis on the ongoing development of the transformer model. I'm actively working on enhancing its functionality and performance.

# Motivation
I initiated this project a few months ago with a clear goal: to develop an offline ChatGPT-like Natural Language Model capable of training with my own data. Along the way, I've learned a lot about dataset processing, word embeddings, language model architectures, and the powerful role of linear algebra in understanding human language.

Through numerous iterations and diverse architectures, I've made significant progress in reaching my current implementation. However, the project remains a work in progress, constantly evolving and refining.

Most importantly though, I find immense joy and fulfillment in this project. As each day brings me closer to realizing my initial goal, and teaches me valuable information along the way.

# Features
- Dark/Light mode
- Saves the training progress and allows to continue training at any time
- Choose a batch size to allow training on various memory sizes
- Progress bar, descriptive logs and message boxes
- Stats section that shows how many documents are loaded, and how much the model is trained
- Specify the number of documents to load
- Fast training on the GPU

# Dependencies
- Python (3.11.3)
- PyQt6
- PyTorch (CUDA 11.8)
- Gensim
- Numpy
- PyEnchant
- Python-Docx

# How to get started
- Install the newest version of Python (make sure to add it to PATH)
- Install all the dependencies listed above
- Download and extract the repository
- Run 'OLM GUI.py'

## How to process your training data
- Prepare a folder with your training data which contains .docx, .txt, and/or a .json dump called 'story-dump.json'
- Select your training data folder and check 'Load story dump' if you have a .json dump in your training folder
- Optionally, select 'Remove misspelled words' if you want to only allow valid English sentences (Warning: This will remove any sentences containing names, as they are not in the English dictionary)
- Select the number of documents to load if you have a really large dataset (0 will load all documents)
- Press 'Process Training Data', this will clean and tokenize all the documents and create word embeddings with positional encodings

## How to train the model (still WIP)
- Select the desired training parameters
- Select whether to use the GPU for training or not (this requires a NVIDIA GPU with plenty of VRAM and CUDA 11.8 to be installed)
- Press 'Train model' which will start the training process (depending on the size of the training data this might take a long time)

## How to generate text
(As long as the transformer model is under construction, this section only allows interacting with the word embedding model.)

### Getting word synonyms
- Enter a word in the query field (it has to be in the training data)
- Select the number of synonyms you want to get
- Set the temperature selector to a value other than 0.0
- Press 'Generate Text' which will show the most similar words to the one you entered based on the training data
### Getting alternate sentences
- Enter a full sentence or text
- Set the temperature selector to 0.0
- Press 'Generate Text' which will generate an alternate text by replacing almost every word with the one most similar to it
