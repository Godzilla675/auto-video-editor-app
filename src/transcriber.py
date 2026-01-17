import whisper
import warnings
import os

class Transcriber:
    def __init__(self, model_size="base"):
        """
        Initialize the Transcriber with a specific Whisper model size.
        """
        print(f"Loading Whisper model: {model_size}...")
        self.model = whisper.load_model(model_size)
        print("Model loaded.")

    def transcribe(self, video_path):
        """
        Transcribe the audio from the video file.
        Returns the full result dictionary from Whisper.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"Transcribing {video_path}...")
        # Suppress FP16 warning if running on CPU
        warnings.filterwarnings("ignore")
        
        # verbose=False to keep logs clean
        result = self.model.transcribe(video_path, verbose=False)
        print("Transcription complete.")
        return result