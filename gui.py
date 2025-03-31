from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import cv2
import speech
import detect
import os
from gemini_vision import GeminiVision

class HelloWorker(QObject):
    hello_detected = pyqtSignal()
    speech_detected = pyqtSignal(str)

    def __init__(self, engine, project_id):
        super().__init__()
        self.engine = engine
        self.project_id = project_id
        self.running = True

    def run(self):
        while self.running:
            resp = self.engine.recognize_speech_from_mic()
            if resp:
                self.speech_detected.emit(resp)
                intent, text = detect.detect_intent_texts(self.project_id, 0, [resp], 'en')
                if intent == 'Hello':
                    self.running = False
                    self.hello_detected.emit()

class ListenWorker(QObject):
    finished = pyqtSignal()
    result = pyqtSignal(tuple)

    def __init__(self, engine, project_id):
        super().__init__()
        self.engine = engine
        self.project_id = project_id
        self.running = True

    def run(self):
        self.engine.text_speech("Listening")
        while self.running:
            resp = self.engine.recognize_speech_from_mic()
            if resp:
                intent, text = detect.detect_intent_texts(
                    self.project_id, 
                    0, 
                    [resp], 
                    'en'
                )
                self.result.emit((intent, text))
                if intent == 'endconvo':
                    self.running = False
                    break
        self.finished.emit()

class VisionCompanionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Companion")
        # Set a larger window size for better visibility
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: white;
            }
            QTextEdit {
                background-color: #1b1b1b;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
        """)
        
        # Initialize backend
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "api-key.json"
        self.project_id = "soundscape-454816"
        self.engine = speech.speech_to_text()
        self.gemini = GeminiVision()
        
        self.init_ui()
        self.init_camera()
        self.wait_for_hello()

    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left side: Camera feed
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(800, 600)
        self.camera_label.setStyleSheet("border: 2px solid #555; border-radius: 10px; padding: 5px;")
        layout.addWidget(self.camera_label)

        # Right side: Conversation output
        self.conversation = QTextEdit()
        self.conversation.setReadOnly(True)
        self.conversation.setMinimumWidth(400)
        layout.addWidget(self.conversation)

    def init_camera(self):
        self.camera = cv2.VideoCapture(0)
        # Set camera resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Start camera update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)

    def update_camera(self):
        ret, frame = self.camera.read()
        if ret:
            # Convert frame to RGB and resize for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_image = qt_image.scaled(self.camera_label.size(), Qt.KeepAspectRatio)
            self.camera_label.setPixmap(QPixmap.fromImage(scaled_image))

    def log_message(self, message, sender="System"):
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        formatted_message = f"[{timestamp}] {sender}: {message}"
        self.conversation.append(formatted_message)
        # Auto-scroll to bottom
        self.conversation.verticalScrollBar().setValue(
            self.conversation.verticalScrollBar().maximum()
        )

    def wait_for_hello(self):
        self.log_message("Waiting for you to say 'Hello' to start...")
        self.hello_thread = QThread()
        self.hello_worker = HelloWorker(self.engine, self.project_id)
        self.hello_worker.moveToThread(self.hello_thread)
        
        self.hello_thread.started.connect(self.hello_worker.run)
        self.hello_worker.hello_detected.connect(self.start_conversation)
        self.hello_worker.speech_detected.connect(
            lambda text: self.log_message(text, "You")
        )
        
        self.hello_thread.start()

    def start_conversation(self):
        self.log_message("Hello detected! Starting conversation...")
        self.engine.text_speech("What can I help you with?")
        self.log_message("What can I help you with?", "Assistant")
        self.start_listening()

    def start_listening(self):
        self.listen_thread = QThread()
        self.listen_worker = ListenWorker(self.engine, self.project_id)
        self.listen_worker.moveToThread(self.listen_thread)
        
        self.listen_thread.started.connect(self.listen_worker.run)
        self.listen_worker.result.connect(self.handle_command)
        
        self.listen_thread.start()

    def handle_command(self, command_data):
        intent, text = command_data
        self.log_message(text, "You")
        
        if intent == 'Describe':
            ret, frame = self.camera.read()
            if ret:
                description = self.gemini.get_description(frame)
                self.log_message(description, "Assistant")
                self.engine.text_speech(description)

    def closeEvent(self, event):
        self.camera.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = VisionCompanionGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()