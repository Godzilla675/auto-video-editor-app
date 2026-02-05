import requests
import os
from PIL import Image
import io
import time
from typing import Optional

class Generator:
    """
    A class to generate images using the Hugging Face API (FLUX.1-dev model).
    """

    def __init__(self, api_token: str, output_dir: str = "generated_images") -> None:
        """
        Initialize the Generator.

        Args:
            api_token (str): The Hugging Face API token.
            output_dir (str): The directory to save generated images. Defaults to "generated_images".
        """
        self.api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
        self.headers = {"Authorization": f"Bearer {api_token}"}
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate(self, prompt: str, filename_prefix: str = "img") -> Optional[str]:
        """
        Generates an image based on the provided prompt.

        Args:
            prompt (str): The text prompt for image generation.
            filename_prefix (str): Prefix for the saved image filename. Defaults to "img".

        Returns:
            Optional[str]: The file path of the saved image, or None if generation fails.
        """
        print(f"Generating image for prompt: {prompt}")
        payload = {"inputs": prompt}
        
        # Retry logic for model loading
        max_retries = 5
        for i in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload)

                if response.status_code == 200:
                    try:
                        image_bytes = response.content
                        image = Image.open(io.BytesIO(image_bytes))
                        # Use a timestamp for unique filename
                        timestamp = int(time.time())
                        # Sanitize prompt for filename
                        safe_prompt = "".join([c if c.isalnum() else "_" for c in prompt[:20]])
                        filepath = os.path.join(self.output_dir, f"{filename_prefix}_{timestamp}_{safe_prompt}.png")
                        image.save(filepath)
                        print(f"Image saved to {filepath}")
                        return filepath
                    except Exception as e:
                        print(f"Error saving image: {e}")
                        return None
                elif response.status_code == 503:
                    # Model loading
                    try:
                        wait_time = response.json().get('estimated_time', 20)
                    except Exception:
                        wait_time = 20
                    print(f"Model loading... retrying in {wait_time} seconds")
                    time.sleep(wait_time)
                else:
                    print(f"Error generating image: {response.status_code} - {response.text}")
                    return None
            except requests.RequestException as e:
                print(f"Request failed: {e}")
                return None
        
        return None
