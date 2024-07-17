import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal
import warnings
warnings.filterwarnings("ignore")


def load_model_and_processor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning")
    return image_processor, tokenizer, model, device


def extract_frames(video_path, model):
    container = av.open(video_path)
    seg_len = container.streams.video[0].frames
    clip_len = model.config.encoder.num_frames
    indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return frames


def generate_caption(video_path):
    image_processor, tokenizer, model, device = load_model_and_processor()
    frames = extract_frames(video_path, model)
    gen_kwargs = {
        "min_length": 30, # minimum len of descripton generated
        "max_length": 150, # maximum len of descripton generated
        "num_beams": 8,
        "no_repeat_ngram_size": 2,
        "early_stopping": True,
    }
    print("step 1: video processor loaded")
    pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
    print("step 2: text processor loaded")
    tokens = model.generate(pixel_values, **gen_kwargs)
    print("Caption :- ")
    caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    print(caption)
    return caption

class CaptionThread(QThread):
    update_caption = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        # You might want to handle exceptions here
        caption = generate_caption(self.video_path)
        self.update_caption.emit(caption)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        # Text field for video path
        self.path_entry = QLineEdit(self)
        self.path_entry.setPlaceholderText("Enter the path to a video file or browse to select...")
        self.layout.addWidget(self.path_entry)

        # Button to browse for file
        self.browse_button = QPushButton('Browse', self)
        self.browse_button.clicked.connect(self.openFileDialog)
        self.layout.addWidget(self.browse_button)

        # Label for captions
        self.caption_label = QLabel('Caption will appear here...', self)
        self.layout.addWidget(self.caption_label)

        # Button to generate captions
        self.generate_button = QPushButton('Generate Caption', self)
        self.generate_button.clicked.connect(self.generateCaption)
        self.layout.addWidget(self.generate_button)

        self.setLayout(self.layout)
        self.setWindowTitle('Video Caption Generator')

    def openFileDialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "Video Files (*.mp4 *.avi);;All Files (*)", options=options)
        if file_name:
            self.path_entry.setText(file_name)

    def generateCaption(self):
        video_path = self.path_entry.text()
        if video_path:
            self.caption_label.setText('Loading...')
            self.caption_thread = CaptionThread(video_path)
            self.caption_thread.update_caption.connect(self.updateCaption)
            self.caption_thread.start()

    def updateCaption(self, caption):
        self.caption_label.setText(caption)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())

