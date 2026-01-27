import argparse
import os
import requests
import json
import shutil
import concurrent.futures
from typing import Optional, Dict, Any

from src.transcriber import Transcriber
from src.analyzer import Analyzer
from src.generator import Generator
from src.editor import Editor

def download_video(url: str, filename: str = "input.mp4") -> Optional[str]:
    """
    Downloads a video from a URL to a local file.

    Args:
        url (str): The URL of the video to download.
        filename (str): The local filename to save the video as. Defaults to "input.mp4".

    Returns:
        Optional[str]: The path to the downloaded file, or None if download fails.
    """
    print(f"Downloading video from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        print("Download complete.")
        return filename
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def main() -> None:
    """
    Main execution function for the Automated Video Editor.
    """
    parser = argparse.ArgumentParser(description="Automated Video Editor")
    parser.add_argument("--video", help="Path to video file", default=None)
    parser.add_argument("--url", help="URL of video to download", default=None)
    parser.add_argument("--output", help="Output filename", default="final_video.mp4")
    
    # New features arguments
    parser.add_argument("--music", help="Path or URL to background music file", default=None)
    parser.add_argument("--music-volume", help="Volume of background music (0.0-1.0)", type=float, default=0.1)
    parser.add_argument("--crossfade", help="Crossfade duration in seconds", type=float, default=0.0)
    parser.add_argument("--filter", help="Visual filter to apply (black_white, invert_colors, painting, mirror_x, mirror_y)", default=None)

    # Intro/Outro arguments
    parser.add_argument("--intro-text", help="Text to display at the start of the video", default=None)
    parser.add_argument("--outro-text", help="Text to display at the end of the video", default=None)

    # Subtitle arguments
    parser.add_argument("--font", help="Subtitle font", default="DejaVuSans")
    parser.add_argument("--fontsize", help="Subtitle font size", type=int, default=40)
    parser.add_argument("--color", help="Subtitle color", default="white")
    parser.add_argument("--stroke_color", help="Subtitle stroke color", default="black")
    parser.add_argument("--stroke_width", help="Subtitle stroke width", type=int, default=2)

    args = parser.parse_args()
    
    # 1. Input handling
    video_path: Optional[str] = args.video
    if args.url:
        video_path = download_video(args.url)

    # Handle music download if URL
    music_path = args.music
    if music_path and (music_path.startswith("http://") or music_path.startswith("https://")):
         print("Downloading background music...")
         music_path = download_video(music_path, filename="background_music.mp3")
    
    if not video_path or not os.path.exists(video_path):
        print("Error: No valid video file provided.")
        return

    # 2. Setup Components
    gemini_key = os.environ.get("GEMINI_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    
    if not gemini_key or not hf_token:
        print("Error: GEMINI_API_KEY and HF_TOKEN environment variables must be set.")
        return

    transcriber = Transcriber()
    analyzer = Analyzer(api_key=gemini_key)
    generator = Generator(api_token=hf_token)
    editor = Editor()

    # 3. Transcribe
    print("\n--- Step 1: Transcription ---")
    try:
        transcription_result = transcriber.transcribe(video_path)
        # Simplify transcription for analysis (just text and segments)
        transcription_data: Dict[str, Any] = {
            "text": transcription_result["text"],
            "segments": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in transcription_result["segments"]]
        }
    except Exception as e:
        print(f"Transcription failed: {e}")
        return

    # 4. Analyze
    print("\n--- Step 2: Analysis ---")
    analysis_data = analyzer.analyze(video_path, transcription_data)
    if not analysis_data:
        print("Analysis failed. Exiting.")
        return
    
    print("Analysis Result:")
    print(json.dumps(analysis_data, indent=2))

    # 5. Generate Graphics
    print("\n--- Step 3: Graphics Generation ---")
    graphic_paths: Dict[int, str] = {}
    graphics_reqs = analysis_data.get("graphics", [])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_index = {}
        for i, req in enumerate(graphics_reqs):
            prompt = req.get("prompt")
            if prompt:
                future = executor.submit(generator.generate, prompt, filename_prefix=f"graphic_{i}")
                future_to_index[future] = i

        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            try:
                path = future.result()
                if path:
                    graphic_paths[i] = path
            except Exception as e:
                print(f"Error generating graphic {i}: {e}")

    # 6. Edit
    print("\n--- Step 4: Editing ---")

    subtitle_config = {
        "font": args.font,
        "fontsize": args.fontsize,
        "color": args.color,
        "stroke_color": args.stroke_color,
        "stroke_width": args.stroke_width
    }

    output_path = editor.edit(video_path, analysis_data, graphic_paths, output_path=args.output,
                              music=music_path, music_volume=args.music_volume, crossfade=args.crossfade,
                              subtitle_config=subtitle_config, visual_filter=args.filter,
                              intro_text=args.intro_text, outro_text=args.outro_text)
    
    if output_path:
        print(f"\nSuccess! Final video available at: {output_path}")
    else:
        print("\nEditing failed.")

if __name__ == "__main__":
    main()
