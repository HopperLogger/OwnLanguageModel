import json
import sys
import os
from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QPushButton, QCheckBox, QProgressBar, QFileDialog, QSpinBox, QFrame, QHBoxLayout, QVBoxLayout, QLineEdit, QDoubleSpinBox, QTextEdit
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon
from PyQt6.QtCore import Qt

try:
    import ProcessTrainingData
    import TrainModel
    import GenerateText
except:
    print('ERROR: Not all dependencies are present!')

class MainWindow(QWidget):

    # Parameters of the GUI
    big_font = QFont('Segoe UI', 16)
    big_font.setBold(True)
    font = QFont('Segoe UI', 14)
    log_font = QFont('Segoe UI', 13)
    small_font = QFont('Segoe UI', 12)
    global_progress = 0

    # Define a light color palette
    light_palette = QPalette()
    light_palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    light_palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
    light_palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    light_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
    light_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    light_palette.setColor(QPalette.ColorRole.ToolTipText,QColor(0, 0, 0))
    light_palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
    light_palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
    light_palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
    light_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    light_palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(100, 100, 100))

    # Define a dark color palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(40, 40, 40))
    dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(200, 0, 0))
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(150, 150, 150))

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.process_training_data_thread = None
        self.train_model_thread = None
        self.generate_text_thread = None
        # ----------------------- PROCESSING SECTION --------------------------------------------
        # Create the processing section label
        self.processing_section_label = QLabel("Processing:")
        self.processing_section_label.setFont(self.big_font)
        
        # Create the paths section label
        self.paths_section_label = QLabel("Paths:")
        self.paths_section_label.setFont(self.font)
        
        # Create the training folder browser
        self.selected_folder_label = QLineEdit()
        self.selected_folder_label.setPlaceholderText('No training folder selected')
        self.folder_browse_button = QPushButton('Browse')
        self.folder_browse_button.clicked.connect(self.openFolderDialog)
        hbox_folder_browse = QHBoxLayout()
        hbox_folder_browse.addWidget(self.selected_folder_label)
        hbox_folder_browse.addWidget(self.folder_browse_button)

        # Create the embedding file browser
        self.selected_file_label = QLineEdit()
        self.selected_file_label.setPlaceholderText('No pre-trained embedding selected')
        self.file_browse_button = QPushButton('Browse')
        self.file_browse_button.clicked.connect(self.openFileDialog)
        hbox_file_browse = QHBoxLayout()
        hbox_file_browse.addWidget(self.selected_file_label)
        hbox_file_browse.addWidget(self.file_browse_button)

        # Create the checkbox for loading the story dump
        self.load_story_dump_cb = QCheckBox("Load story dump")
        self.load_story_dump_cb.setFont(self.small_font)
        self.load_story_dump_cb.setChecked(True)
        self.load_story_dump_cb.setToolTip('Load the story dump from the training folder as created by the Website Downloader.')

        # Create the checkbox for removing misspelled words
        self.check_spelling_cb = QCheckBox("Remove misspelled words")
        self.check_spelling_cb.setFont(self.small_font)
        self.check_spelling_cb.setChecked(False)
        self.check_spelling_cb.setToolTip('Remove misspelled and not English words from the training data. This will take a long time and will remove most names.')

        # Create the samples to keep label
        self.samples_to_keep_label = QLabel("Number of documents to load:")
        self.samples_to_keep_label.setFont(self.small_font)

        # Create the samples to keep selector
        self.num_samples_to_keep_selector = QSpinBox()
        self.num_samples_to_keep_selector.setFixedSize(300, 30)
        self.num_samples_to_keep_selector.setFont(self.small_font)
        self.num_samples_to_keep_selector.setMinimum(0)
        self.num_samples_to_keep_selector.setMaximum(1000000)
        self.num_samples_to_keep_selector.setValue(0)
        self.num_samples_to_keep_selector.setToolTip('Number of documents to process. Set to 0 to load all documents.')
        
        # Create the process training data button
        self.process_training_data_bt = QPushButton("Process Training Data")
        self.process_training_data_bt.setFixedSize(220, 50)
        self.process_training_data_bt.setFont(self.small_font)
        self.process_training_data_bt.clicked.connect(self.createProcessTrainingDataThread)
        self.process_training_data_bt.setToolTip('Process the training data.')

        vbox_processing_pars = QVBoxLayout()
        vbox_processing_pars.addWidget(self.load_story_dump_cb)
        vbox_processing_pars.addWidget(self.check_spelling_cb)
        vbox_processing_pars.addWidget(self.samples_to_keep_label)
        vbox_processing_pars.addWidget(self.num_samples_to_keep_selector)
        vbox_processing_pars.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Create an hbox for the processing section
        hbox_processing_section = QHBoxLayout()
        hbox_processing_section.addLayout(vbox_processing_pars)
        hbox_processing_section.addSpacing(40)
        hbox_processing_section.addWidget(self.process_training_data_bt)
        hbox_processing_section.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # ----------------------- TRAINING SECTION --------------------------------------------
        # Create the training section label
        self.training_section_label2 = QLabel("Training:")
        self.training_section_label2.setFont(self.big_font)
        # Create the epoch selector
        self.num_epochs_selector = QSpinBox()
        self.num_epochs_selector.setFixedSize(300, 30)
        self.num_epochs_selector.setFont(self.small_font)
        self.num_epochs_selector.setMinimum(1)
        self.num_epochs_selector.setMaximum(1000)
        self.num_epochs_selector.setValue(30)
        self.num_epochs_selector.setToolTip('Number of steps to train the model. More is better, but takes longer.')

        # Create the batch size selector
        self.num_batch_selector = QSpinBox()
        self.num_batch_selector.setFixedSize(300, 30)
        self.num_batch_selector.setFont(self.small_font)
        self.num_batch_selector.setMinimum(1)
        self.num_batch_selector.setMaximum(2048)
        self.num_batch_selector.setValue(32)
        self.num_batch_selector.setToolTip('Number of samples to process at once. More is better, but takes longer.')

        # Create the ltsm selector
        self.num_ltsm_selector = QSpinBox()
        self.num_ltsm_selector.setFixedSize(300, 30)
        self.num_ltsm_selector.setFont(self.small_font)
        self.num_ltsm_selector.setMinimum(64)
        self.num_ltsm_selector.setMaximum(512)
        self.num_ltsm_selector.setValue(128)
        self.num_ltsm_selector.setToolTip('Number of nodes in the LSTM layer. More is better, but takes longer.')

        # Create the training test size selector
        self.num_training_test_size_selector = QDoubleSpinBox()
        self.num_training_test_size_selector.setFixedSize(300, 30)
        self.num_training_test_size_selector.setFont(self.small_font)
        self.num_training_test_size_selector.setMinimum(0.0001)
        self.num_training_test_size_selector.setMaximum(1)
        self.num_training_test_size_selector.setValue(0.01)
        self.num_training_test_size_selector.setSingleStep(0.0001)
        self.num_training_test_size_selector.setDecimals(4)
        self.num_training_test_size_selector.setToolTip('Percentage of the training data to use for testing.')

        # Create the learning rate selector
        self.num_learning_rate_selector = QDoubleSpinBox()
        self.num_learning_rate_selector.setFixedSize(300, 30)
        self.num_learning_rate_selector.setFont(self.small_font)
        self.num_learning_rate_selector.setMinimum(0.001)
        self.num_learning_rate_selector.setMaximum(1)
        self.num_learning_rate_selector.setValue(0.001)
        self.num_learning_rate_selector.setSingleStep(0.001)
        self.num_learning_rate_selector.setDecimals(5)
        self.num_learning_rate_selector.setToolTip('Learning rate for the model. Smaller is better, but takes longer.')

        # Create the max sequence length selector
        self.num_seq_length_selector = QSpinBox()
        self.num_seq_length_selector.setFixedSize(300, 30)
        self.num_seq_length_selector.setFont(self.small_font)
        self.num_seq_length_selector.setMinimum(8)
        self.num_seq_length_selector.setMaximum(1024)
        self.num_seq_length_selector.setValue(256)
        self.num_seq_length_selector.setToolTip('Maximum length of a sequence. Longer is better, but takes longer.')

        # Create the GPU checkbox
        self.gpu_cb = QCheckBox("Use GPU")
        self.gpu_cb.setFont(self.small_font)
        self.gpu_cb.setChecked(True)
        self.gpu_cb.setToolTip('Use the GPU to train the model. Faster, but requires a compatible GPU.')

        # Create the train model button
        self.train_model_bt = QPushButton("Train Model")
        self.train_model_bt.setFixedSize(220, 50)
        self.train_model_bt.setFont(self.small_font)
        self.train_model_bt.clicked.connect(self.createTrainModelThread)
        self.train_model_bt.setToolTip('Train the model with the selected parameters.')

        # Create the abort button
        self.abort_bt = QPushButton("Abort")
        self.abort_bt.setFixedSize(110, 25)
        self.abort_bt.setFont(self.small_font)
        self.abort_bt.clicked.connect(self.abortThread)
        self.abort_bt.setStyleSheet("background-color: #805050; color: #ffffff;")

        # Create the epoch label
        self.epoch_label = QLabel("Number of epochs:")
        self.epoch_label.setFont(self.small_font)

        # Create the batch size label
        self.batch_size_label = QLabel("Batch size:")
        self.batch_size_label.setFont(self.small_font)

        # Create the ltsm label
        self.ltsm_label = QLabel("LSTM size:")
        self.ltsm_label.setFont(self.small_font)

        # Create the training test size label
        self.training_test_size_label = QLabel("Training test size:")
        self.training_test_size_label.setFont(self.small_font)

        # Create the learning rate label
        self.learning_rate_label = QLabel("Learning rate:")
        self.learning_rate_label.setFont(self.small_font)

        # Create the max sequence length label
        self.seq_length_label = QLabel("Max sequence length:")
        self.seq_length_label.setFont(self.small_font)

        # Create the layout for the training parameters
        self.vboxpars1 = QVBoxLayout()
        self.vboxpars1.addWidget(self.epoch_label)
        self.vboxpars1.addWidget(self.num_epochs_selector)
        self.vboxpars1.addWidget(self.batch_size_label)
        self.vboxpars1.addWidget(self.num_batch_selector)
        self.vboxpars1.addWidget(self.ltsm_label)
        self.vboxpars1.addWidget(self.num_ltsm_selector)
        self.vboxpars1.addSpacing(30)
        self.vboxpars1.addWidget(self.gpu_cb, alignment=Qt.AlignmentFlag.AlignCenter)
        self.vboxpars1.addStretch()
        self.vboxpars2 = QVBoxLayout()
        self.vboxpars2.addWidget(self.learning_rate_label)
        self.vboxpars2.addWidget(self.num_learning_rate_selector)
        self.vboxpars2.addWidget(self.seq_length_label)
        self.vboxpars2.addWidget(self.num_seq_length_selector)
        self.vboxpars2.addWidget(self.training_test_size_label)
        self.vboxpars2.addWidget(self.num_training_test_size_selector)
        self.vboxpars2.addSpacing(20)
        self.vboxpars2.addWidget(self.train_model_bt, alignment=Qt.AlignmentFlag.AlignCenter)
        self.vboxpars2.addStretch()

        self.hboxpars = QHBoxLayout()
        self.hboxpars.addLayout(self.vboxpars1)
        self.hboxpars.addLayout(self.vboxpars2)
        self.hboxpars.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # ----------------------- LOGS SECTION --------------------------------------------
        # Create the logs section label
        logs_label = QLabel("Logs:")
        logs_label.setFont(self.font)

        # Create the log box
        self.log_box = QTextEdit()
        self.log_box.setFont(self.log_font)
        self.log_box.setReadOnly(True)
        self.log_box.setFixedSize(650, 200)

        # Create the progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedSize(685, 30)

        # Create a horizontal line to seperate the two sections
        self.hline = QFrame(self)
        self.hline.setFrameShape(QFrame.Shape.HLine)
        self.hline.setFrameShadow(QFrame.Shadow.Sunken)
        self.hline.setStyleSheet("color: #ffffff")

        # Create the layout for the processing and training section
        vbox_left = QVBoxLayout()
        vbox_left.addWidget(self.processing_section_label)
        vbox_left.addWidget(self.paths_section_label)
        vbox_left.addLayout(hbox_folder_browse)
        vbox_left.addLayout(hbox_file_browse)
        vbox_left.addWidget(self.processing_section_label)
        vbox_left.addLayout(hbox_processing_section)
        vbox_left.addWidget(self.hline)
        vbox_left.addWidget(self.training_section_label2)
        vbox_left.addLayout(self.hboxpars)
        vbox_left.addWidget(logs_label)
        vbox_left.addWidget(self.log_box)
        vbox_left.addWidget(self.progress_bar)

        # ----------------------- TEXT GENERATION SECTION --------------------------------------------
        # Create the text generation section label
        text_generation_label = QLabel("Text Generation:")
        text_generation_label.setFont(self.big_font)

        # Create the query label
        query_label = QLabel("Query:")
        query_label.setFont(self.font)

        # Create entry field for the query
        self.query_entry_box = QLineEdit()
        self.query_entry_box.setFixedSize(300, 30)
        self.query_entry_box.setFont(self.small_font)
        self.query_entry_box.setPlaceholderText("Enter an input query.")

        # Create counter for the number of words
        self.num_words_selector = QSpinBox()
        self.num_words_selector.setFixedSize(300, 30)
        self.num_words_selector.setFont(self.small_font)
        self.num_words_selector.setMinimum(1)
        self.num_words_selector.setMaximum(1000)
        self.num_words_selector.setValue(10)
        self.num_words_selector.setToolTip("Select the number of words to generate.")

        # Create label for the number of words
        self.num_words_label = QLabel("Number of words:")
        self.num_words_label.setFont(self.font)

        # Create label for the temperature
        self.temperature_label = QLabel("Temperature:")
        self.temperature_label.setFont(self.font)

        # Create counter for the temperature
        self.num_temperature_selector = QDoubleSpinBox()
        self.num_temperature_selector.setFixedSize(300, 30)
        self.num_temperature_selector.setFont(self.small_font)
        self.num_temperature_selector.setMinimum(0.0001)
        self.num_temperature_selector.setMaximum(1)
        self.num_temperature_selector.setValue(0.5)
        self.num_temperature_selector.setSingleStep(0.1)
        self.num_temperature_selector.setToolTip("Select the temperature for the generation. The higher the temperature, the more random the generation.")

        # Create label for the generated text
        generated_text_label = QLabel("Generated Text:")
        generated_text_label.setFont(self.font)

        # Create the text box for the generated text
        self.generated_text_box = QTextEdit()
        self.generated_text_box.setFixedSize(300, 300)
        self.generated_text_box.setFont(self.small_font)
        self.generated_text_box.setReadOnly(True)
        self.generated_text_box.setPlaceholderText("Generated text will appear here.")

        # Create the generate text button
        self.generate_text_bt = QPushButton("Generate Text")
        self.generate_text_bt.setFixedSize(200, 50)
        self.generate_text_bt.setFont(self.small_font)
        self.generate_text_bt.clicked.connect(self.createGenerateTextThread)
        self.generate_text_bt.setToolTip("Generate text based on the query.")

        # Create a dark mode button
        self.dark_mode_bt = QPushButton("Light Mode")
        self.dark_mode_bt.setFixedSize(110, 25)
        self.dark_mode_bt.setFont(self.small_font)
        self.dark_mode_bt.setStyleSheet("background-color: #ffffff; color: #000000")
        self.dark_mode_bt.clicked.connect(self.lightMode)
        self.dark_mode_bt.setToolTip("Switches the theme.")

        # Create the stats
        self.model_label = QLabel("Model: OLM V6.1.3")
        self.model_label.setFont(self.small_font)
        self.model_label.setStyleSheet("color: #808080")

        self.training_data_label = QLabel()
        self.training_data_label.setFont(self.small_font)
        self.training_data_label.setStyleSheet("color: #808080")

        self.trained_model_label = QLabel()
        self.trained_model_label.setFont(self.small_font)
        self.trained_model_label.setStyleSheet("color: #808080")

        self.updateStats()

        vbox_stats = QVBoxLayout()
        vbox_stats.addWidget(self.model_label, alignment=Qt.AlignmentFlag.AlignCenter)
        vbox_stats.addWidget(self.training_data_label, alignment=Qt.AlignmentFlag.AlignCenter)
        vbox_stats.addWidget(self.trained_model_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Create the layout for the options
        hbox_options = QHBoxLayout()
        hbox_options.addWidget(self.abort_bt, alignment=Qt.AlignmentFlag.AlignRight)
        hbox_options.addWidget(self.dark_mode_bt, alignment=Qt.AlignmentFlag.AlignRight)

        # Create the layout for the text generation section
        vbox_right = QVBoxLayout()
        vbox_right.addWidget(text_generation_label)
        vbox_right.addWidget(query_label)
        vbox_right.addWidget(self.query_entry_box)
        vbox_right.addWidget(self.num_words_label)
        vbox_right.addWidget(self.num_words_selector)
        vbox_right.addWidget(self.temperature_label)
        vbox_right.addWidget(self.num_temperature_selector)
        vbox_right.addStretch()
        vbox_right.addWidget(generated_text_label)
        vbox_right.addWidget(self.generated_text_box)
        vbox_right.addSpacing(10)
        vbox_right.addWidget(self.generate_text_bt, alignment=Qt.AlignmentFlag.AlignCenter)
        vbox_right.addSpacing(30)
        vbox_right.addLayout(vbox_stats)
        vbox_right.addSpacing(50)
        vbox_right.addLayout(hbox_options)

        # ----------------------- WINDOW LAYOUT --------------------------------------------
        # Create a vertical line to seperate the two sections
        vline = QFrame(self)
        vline.setFrameShape(QFrame.Shape.VLine)
        vline.setFrameShadow(QFrame.Shadow.Sunken)
        
        hbox_window = QHBoxLayout()
        hbox_window.addLayout(vbox_left)
        hbox_window.addWidget(vline)
        hbox_window.addLayout(vbox_right)
                
        # Set the layout of the window
        self.setLayout(hbox_window)

        # Set the size and title of the window
        #self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setGeometry(400, 100, 1000, 600)
        self.setWindowTitle('Own Language Model GUI')
        self.darkMode()

    # ----------------------- WINDOW FUNCTIONS ------------------------------------------
    # Function to open a folder dialog and get the selected folder path
    def openFolderDialog(self):
            # Create a QFileDialog widget
            file_dialog = QFileDialog(None, "Select Training Folder", options=QFileDialog.Option.ShowDirsOnly)
            file_dialog.setFileMode(QFileDialog.FileMode.Directory)

            # Run the file dialog and get the selected file path
            if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                selected_folder = file_dialog.selectedFiles()[0]
                self.selected_folder_label.setText(selected_folder)
                self.training_folder = selected_folder

    # Function to open a file dialog and get the selected file path
    def openFileDialog(self):
            # Create a QFileDialog widget
            file_dialog = QFileDialog(None, "Select Embedding File")

            # Run the file dialog and get the selected file path
            if file_dialog.exec():
                selected_file = file_dialog.selectedFiles()[0]
                self.selected_file_label.setText(selected_file)
                self.embedding_path = selected_file

    def darkMode(self):
        self.setPalette(self.dark_palette)
        # Iterate over all widgets in the main window
        for widget in self.findChildren(QLineEdit):
            widget.setPalette(self.dark_palette)
        self.dark_mode_bt.setText("Light Mode")
        self.dark_mode_bt.setStyleSheet("background-color: #ffffff; color: #000000")
        self.dark_mode_bt.clicked.connect(self.lightMode)
        self.folder_browse_button.setStyleSheet("background-color: #505050; color: #ffffff")
        self.file_browse_button.setStyleSheet("background-color: #505050; color: #ffffff")
        self.process_training_data_bt.setStyleSheet("background-color: #505050; color: #ffffff")
        self.train_model_bt.setStyleSheet("background-color: #505050; color: #ffffff")
        self.generate_text_bt.setStyleSheet("background-color: #505050; color: #ffffff")
        self.process_training_data_bt.setStyleSheet("background-color: #505080; color: #ffffff")
        self.train_model_bt.setStyleSheet("background-color: #807050; color: #ffffff")
        self.generate_text_bt.setStyleSheet("background-color: #508050; color: #ffffff")

    def lightMode(self):
        self.setPalette(self.light_palette)
        # Iterate over all widgets in the main window
        for widget in self.findChildren(QLineEdit):
            widget.setPalette(self.light_palette)
        self.dark_mode_bt.setText("Dark Mode")
        self.dark_mode_bt.setStyleSheet("background-color: #505050; color: #ffffff")
        self.dark_mode_bt.clicked.connect(self.darkMode)
        self.folder_browse_button.setStyleSheet("")
        self.file_browse_button.setStyleSheet("")
        self.process_training_data_bt.setStyleSheet("")
        self.train_model_bt.setStyleSheet("")
        self.generate_text_bt.setStyleSheet("")

    # Starts the process training data thread
    def createProcessTrainingDataThread(self):
        # Grab the values from the GUI
        self.load_stories = self.load_story_dump_cb.isChecked()
        self.check_spelling = self.check_spelling_cb.isChecked()
        self.samples_to_keep = self.num_samples_to_keep_selector.value()
        self.training_folder = self.selected_folder_label.text()
        self.embedding_path = self.selected_file_label.text()
        self.epochs = self.num_epochs_selector.value()

        # Create a new thread and connect the progress bar and log message to the corresponding methods
        self.process_training_data_thread = ProcessTrainingData.ProcessTrainingData(self.training_folder, self.embedding_path, self.load_stories, 
                                                                                    self.samples_to_keep, self.check_spelling, self.epochs)
        self.process_training_data_thread.increaseProgressBar.connect(self.increaseProgressBar)
        self.process_training_data_thread.sendLogMessage.connect(self.updateLogs)
        self.process_training_data_thread.updateStats.connect(self.updateStats)

        # Start the thread
        self.progress_bar.setValue(0)
        self.log_box.clear()
        self.process_training_data_thread.start()

    # Starts the train model thread
    def createTrainModelThread(self):
        # Grab the values from the GUI
        self.epochs = self.num_epochs_selector.value()
        self.batch_size = self.num_batch_selector.value()
        self.ltsm = self.num_ltsm_selector.value()
        self.training_test_size = self.num_training_test_size_selector.value()
        self.learning_rate = self.num_learning_rate_selector.value()
        self.seq_length = self.num_seq_length_selector.value()
        self.samples_to_keep = self.num_samples_to_keep_selector.value()
        self.gpu_enable = self.gpu_cb.isChecked()
        # Create a new thread and connect the progress bar and log message to the corresponding methods
        self.train_model_thread = TrainModel.TrainModel(self.epochs, self.batch_size, self.ltsm, 
                                                        self.training_test_size, self.learning_rate, self.seq_length, 
                                                        self.samples_to_keep, self.gpu_enable)
        self.train_model_thread.increaseProgressBar.connect(self.increaseProgressBar)
        self.train_model_thread.sendLogMessage.connect(self.updateLogs)
        self.train_model_thread.updateStats.connect(self.updateStats)

        # Start the thread
        self.progress_bar.setValue(0)
        self.log_box.clear()
        self.train_model_thread.start()

    # Starts the generate text thread
    def createGenerateTextThread(self):
        # Grab the values from the GUI
        self.query_entry = self.query_entry_box.text()
        self.temperature = self.num_temperature_selector.value()
        self.num_words = self.num_words_selector.value()
        self.gpu_enabled = self.gpu_cb.isChecked()

        # Create a new thread and connect the progress bar and log message to the corresponding methods
        self.generate_text_thread = GenerateText.GenerateText(self.query_entry, self.temperature, self.num_words, self.gpu_enabled)
        self.generate_text_thread.increaseProgressBar.connect(self.increaseProgressBar)
        self.generate_text_thread.sendLogMessage.connect(self.updateLogs)
        self.generate_text_thread.setGeneratedText.connect(self.updateGeneratedText)

        # Start the thread
        self.progress_bar.setValue(0)
        self.log_box.clear()
        self.generate_text_thread.start()

    def abortThread(self):
        if self.process_training_data_thread is not None and self.process_training_data_thread.isRunning():
            self.process_training_data_thread.terminate()
            self.updateLogs("Aborted.", "red")
            self.progress_bar.setValue(0)
        if self.train_model_thread is not None and self.train_model_thread.isRunning():
            self.train_model_thread.terminate()
            self.updateLogs("Aborted.", "red")
            self.progress_bar.setValue(0)
            self.updateStats("model")
        if self.generate_text_thread is not None and self.generate_text_thread.isRunning():
            self.generate_text_thread.terminate()
            self.updateLogs("Aborted.", "red")
            self.progress_bar.setValue(0)

    def updateStats(self, part: str = "all") -> None:
        """
        Updates the part of the stats section specified by the parameter.

        Parameters:
        part (str): The part of the stats section to update. Can be "training-data", "model", or "all"

        Returns:
        None
        """

        # Update the training data stats
        if part == "training-data" or part == "all":
            if not os.path.exists(os.getcwd() + '/ProcessedData/training-data-0.json'):
                self.training_data_label.setText("Training Data: Not Available")
            else:
                if not os.path.exists(os.getcwd() + '/ProcessedData/training-data-stats.json'):
                    self.training_data_label.setText("Training Data: Not Available")
                else:
                    with open(os.getcwd() + '/ProcessedData/training-data-stats.json', 'r') as f:
                        num_total_documents = json.load(f)
                        self.training_data_label.setText("Training Data: " + str('{:,}'.format(num_total_documents)) + " documents")

        # Update the model stats
        if part == "model" or part == "all":
            if not os.path.exists(os.getcwd() + '/ProcessedData/word2vec-model-steps.json'):
                self.trained_model_label.setText("Trained Model: Not Available")
            else:
                with open(os.getcwd() + '/ProcessedData/word2vec-model-steps.json', 'r') as f:
                    trained_epochs = json.load(f)
                    self.trained_model_label.setText("Trained Model: " + str(trained_epochs) + " epochs")

    def increaseProgressBar(self, amount: float) -> None:
        """
        Updates the progress bar by incrementing its value by the amount specified.

        Parameters:
        amount (int): The amount that the progress bar should be incremented by.

        Special:
        If amount is 0, the progress bar will be reset to 0.

        Returns:
        None
        """
    
        # If amount is 0, reset the progress bar to 0
        if amount == 0:
            self.progress_bar.setValue(0)
            self.global_progress = 0
            return
        
        # If the progress bar is already at 100%, do nothing
        if self.global_progress >= 100:
            self.progress_bar.setValue(100)
            return

        # Otherwise, increment the progress bar by the amount specified
        old_progress = int(self.global_progress)
        self.global_progress += amount
        new_progress = int(self.global_progress)
        if new_progress > 100:
            new_progress = 100
        if new_progress > old_progress:
            self.progress_bar.setValue(new_progress)

    def updateLogs(self, message: str, color: str = 'white') -> None:
        """
        Updates the log box with a new message.

        Parameters:
        message (str): The new message to be displayed in the log box.
        color (str): The color of the message. Default is white.
        
        Special:
        If message is empty, the log box will be cleared.

        Returns:
        None
        """
        # Clear the log box if the message is empty
        if message == '':
            self.log_box.setText('')
        
        # Set the color of the message
        if color == 'red':
            message = '<span style="color: rgb(255, 0, 0);">' + message + '</span>'
        elif color == 'yellow':
            message = '<span style="color: rgb(200, 160, 0);">' + message + '</span>'
        elif color == 'green':
            message = '<span style="color: rgb(0, 255, 50);">' + message + '</span>'
        elif color == 'blue':
            message = '<span style="color: rgb(0, 200, 255);">' + message + '</span>'

        # If the log box is empty, add the message without the <br> tag
        current_text = self.log_box.toPlainText()
        if current_text == '':
            new_text = message
        else:
            new_text = '<br>' + message
        self.log_box.insertHtml(new_text)
        # Scroll to the bottom of the log box
        scrollbar = self.log_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def updateGeneratedText(self, text: str) -> None:
        """
        Updates the generated text box with the generated text.

        Parameters:
        text (str): The generated text.

        Returns:
        None
        """
        self.generated_text_box.setText(text)

# ----------------------- MAIN FUNCTION ------------------------------------------

if __name__ == '__main__':
    # Clear the terminal
    os.system('cls' if os.name=='nt' else 'clear')

    # Create the application and the main window
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
