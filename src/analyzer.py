import os
import cv2
import base64
import json
import math
from openai import OpenAI

class Analyzer:
    def __init__(self, api_key, base_url="https://godzilla865-cliproxy-api.hf.space/v1", model="gemini-3-pro-preview"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def _extract_frames(self, video_path, num_frames=10):
        """
        Extracts a fixed number of frames from the video, evenly spaced.
        Returns a list of base64 encoded strings.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
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

    def analyze(self, video_path, transcription):
        """
        Analyzes the video and transcription to generate editing instructions.
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

        user_content = []
        user_content.append({"type": "text", "text": f"Here is the transcription of the video:\n{json.dumps(transcription)}"})
        user_content.append({"type": "text", "text": "Here are some frames from the video to help you understand the visual context:"})
        
        for frame in frames:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}"
                }
            })

        print("Sending request to Gemini...")
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
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            print("Failed to parse JSON response. Raw response:")
            print(result_text)
            # Fallback: try to extract JSON from markdown block if present
            if "```json" in result_text:
                try:
                    return json.loads(result_text.split("```json")[1].split("```")[0])
                except:
                    pass
            return None