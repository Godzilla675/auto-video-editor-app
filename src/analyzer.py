import os
import cv2
import base64
import json
import math
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI, APIError

class Analyzer:
    """
    A class to analyze video content and transcription using a Large Language Model (Gemini 3 Pro).
    """

    def __init__(self, api_key: str, base_url: str = "https://godzilla865-cliproxy-api.hf.space/v1", model: str = "gemini-3-pro-preview") -> None:
        """
        Initialize the Analyzer.

        Args:
            api_key (str): The API key for the LLM service.
            base_url (str): The base URL for the LLM API.
            model (str): The model identifier to use.
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def _extract_frames(self, video_path: str, num_frames: int = 10) -> List[str]:
        """
        Extracts a fixed number of frames from the video, evenly spaced.

        Args:
            video_path (str): The path to the video file.
            num_frames (int): The number of frames to extract.

        Returns:
            List[str]: A list of base64 encoded strings representing the extracted frames.

        Raises:
            FileNotFoundError: If the video file is not found.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
             print(f"Error: Could not open video {video_path}")
             return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return []
            
        interval = max(1, total_frames // num_frames)
        frames_base64 = []
        
        print(f"Extracting {num_frames} frames from {video_path}...")

        for i in range(0, total_frames, interval):
            if len(frames_base64) >= num_frames:
                break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Resize to reduce token usage/size (e.g., max 512px width)
                height, width, _ = frame.shape
                max_dim = 512
                if width > max_dim or height > max_dim:
                    scaling_factor = max_dim / float(max(width, height))
                    new_dims = (int(width * scaling_factor), int(height * scaling_factor))
                    frame = cv2.resize(frame, new_dims)

                _, buffer = cv2.imencode('.jpg', frame)
                frames_base64.append(base64.b64encode(buffer).decode('utf-8'))
        
        cap.release()
        return frames_base64

    def analyze(self, video_path: str, transcription: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyzes the video and transcription to generate editing instructions.

        Args:
            video_path (str): Path to the video file.
            transcription (Dict[str, Any]): Transcription data containing text and segments.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing analysis results (segments, captions, graphics, transitions),
                                      or None if analysis fails.
        """
        frames = self._extract_frames(video_path)
        
        # Prepare the prompt
        system_prompt = """
        You are an expert video editor. Your task is to analyze a video and its transcription to create an engaging final cut.
        
        You need to provide a JSON response with the following structure:
        {
            "segments": [{"start": float, "end": float}], // Segments of the video to KEEP. Remove boring or silent parts.
            "captions": [{"start": float, "end": float, "text": string}], // Captions to overlay.
            "graphics": [{"timestamp": float, "prompt": string, "duration": float}], // Moments where a generated graphic is needed to illustrate a concept.
            "transitions": [{"timestamp": float, "type": "crossfade"}] // Suggested transitions.
        }
        
        Ensure the JSON is valid and strictly follows this format. Timestamps are in seconds.
        The "text" in captions should be concise and relevant.
        For "graphics", only suggest them when the speaker is explaining a visual concept that is not shown in the video.
        """

        user_content: List[Dict[str, Any]] = []
        user_content.append({"type": "text", "text": f"Here is the transcription of the video:\n{json.dumps(transcription)}"})
        
        if frames:
            user_content.append({"type": "text", "text": "Here are some frames from the video to help you understand the visual context:"})
            for frame in frames:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame}"
                    }
                })

        print("Sending request to Gemini...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                max_tokens=2000
            )

            result_text = response.choices[0].message.content
            if not result_text:
                print("Error: Empty response from API.")
                return None

            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                print("Failed to parse JSON response. Raw response:")
                print(result_text)
                # Fallback: try to extract JSON from markdown block if present
                if "```json" in result_text:
                    try:
                        return json.loads(result_text.split("```json")[1].split("```")[0])
                    except Exception as e:
                        print(f"Fallback JSON extraction failed: {e}")
                        pass
                elif "```" in result_text:
                     try:
                        return json.loads(result_text.split("```")[1])
                     except Exception as e:
                         print(f"Fallback JSON extraction (no lang) failed: {e}")
                         pass

                # Fallback: search for the first { and last }
                try:
                    start_idx = result_text.find('{')
                    end_idx = result_text.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                         return json.loads(result_text[start_idx:end_idx+1])
                except Exception as e:
                     print(f"Fallback JSON extraction (substring) failed: {e}")
                     pass

                return None

        except APIError as e:
            print(f"OpenAI API Error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during analysis: {e}")
            return None
