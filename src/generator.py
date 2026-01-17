import requests
import os
from PIL import Image
import io
import time

class Generator:
    def __init__(self, api_token, output_dir="generated_images"):
        self.api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
        self.headers = {"Authorization": f"Bearer {api_token}"}
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate(self, prompt, filename_prefix="img"):
        print(f"Generating image for prompt: {prompt}")
        payload = {"inputs": prompt}
        
        # Retry logic for model loading
        max_retries = 5
        for i in range(max_retries):
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                try:
                    image_bytes = response.content
                    image = Image.open(io.BytesIO(image_bytes))
                    # Use a hash or timestamp for unique filename if needed, but prefix is passed
                    timestamp = int(time.time())
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
                print(f"Model loading... retrying in {response.json().get('estimated_time', 20)} seconds")
                time.sleep(response.json().get('estimated_time', 20))
            else:
                print(f"Error generating image: {response.status_code} - {response.text}")
                return None
        
        return None