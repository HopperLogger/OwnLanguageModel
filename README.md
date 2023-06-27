# Own Language Model
My first basic transformer based language model, featuring a GUI for processing the dataset, training the model, and generating text.
This project is still heavily WIP, especially the transformer model is still under heavy construction.

# Dependencies
- PyQt6
- PyTorch
- Gensim
- Numpy
- PyEnchant (only if spellchecking is desired)
- Python-Docx

# How to run the 'Process Training Data' section
- Install the newest version of Python
- Install all the dependencies listed above
- Prepare a folder with your training data which contains .docx, .txt, and/or a .json dump called 'story-dump.json'
- Run 'OLM GUI.py'
- Select your training data folder and select 'Load story dump' if you have a .json dump in your training folder
- Optionally, select 'Remove misspelled words' if you want to only allow valid English sentences (Warning: This will remove any sentences containing names, as they are not in the English dictionary)
- Select the number of documents to load if you have a really large dataset (0 will load all documents)
- Press 'Process Training Data', this will clean and tokenize all the documents and create word embeddings with positional encodings
