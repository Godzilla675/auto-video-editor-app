import whisper
import warnings
import os
from typing import Any, Dict

class Transcriber:
    """
    A class to handle video transcription using OpenAI's Whisper model.
    """

    def __init__(self, model_size: str = "base") -> None:
        """
        Initialize the Transcriber with a specific Whisper model size.

        Args:
            model_size (str): The size of the Whisper model to load (e.g., "base", "small", "medium", "large").
                              Defaults to "base".
        """
        print(f"Loading Whisper model: {model_size}...")
        self.model = whisper.load_model(model_size)
        print("Model loaded.")

    def transcribe(self, video_path: str) -> Dict[str, Any]:
        """
        Transcribe the audio from the video file.

        Args:
            video_path (str): The path to the video file to be transcribed.

        Returns:
            Dict[str, Any]: The full result dictionary from Whisper, containing the transcribed text and segments.

        Raises:
            FileNotFoundError: If the provided video path does not exist.
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