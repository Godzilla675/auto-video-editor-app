from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, ImageClip
import os
from typing import Dict, Any, List, Optional

class Editor:
    """
    A class to handle video editing operations using MoviePy.
    """

    def __init__(self) -> None:
        """
        Initialize the Editor.
        """
        # Configuration for ImageMagick can be done via environment variables
        pass

    def edit(self, video_path: str, analysis_data: Dict[str, Any], graphic_paths: Dict[int, str], output_path: str = "output.mp4") -> Optional[str]:
        """
        Edits the video based on the analysis data and generated graphics.

        Args:
            video_path (str): Path to the source video.
            analysis_data (Dict[str, Any]): Analysis results containing segments, captions, etc.
            graphic_paths (Dict[int, str]): A dictionary mapping graphic indices to file paths.
            output_path (str): Path to save the final video. Defaults to "output.mp4".

        Returns:
            Optional[str]: The path to the output video, or None if editing fails.
        """
        print(f"Editing video: {video_path}")
        try:
            video = VideoFileClip(video_path)
        except Exception as e:
            print(f"Error loading video: {e}")
            return None
        
        segments = analysis_data.get("segments", [])
        if not segments:
            print("No segments defined, using full video.")
            segments = [{"start": 0, "end": video.duration}]
        
        clips = []
        
        # Sort segments by start time to ensure order
        segments.sort(key=lambda x: x["start"])
        
        graphics_reqs = analysis_data.get("graphics", [])
        captions = analysis_data.get("captions", [])

        for seg in segments:
            start = float(seg.get("start", 0))
            end = float(seg.get("end", video.duration))
            
            # Clamp timestamps
            start = max(0, start)
            end = min(video.duration, end)
            
            if start >= end:
                continue
                
            print(f"Processing segment: {start}s - {end}s")
            sub = video.subclip(start, end)
            
            layers = [sub]
            
            # --- Graphics (Overlay) ---
            for i, graphic_req in enumerate(graphics_reqs):
                g_time = float(graphic_req.get("timestamp", 0))
                
                # Check if graphic start point is within this segment
                if start <= g_time < end:
                    img_path = graphic_paths.get(i)
                    if img_path and os.path.exists(img_path):
                        duration = float(graphic_req.get("duration", 3.0))
                        
                        rel_start = g_time - start
                        # Ensure it doesn't exceed segment
                        if rel_start + duration > (end - start):
                            duration = (end - start) - rel_start
                        
                        print(f"Adding graphic {img_path} at relative {rel_start}s")
                        
                        try:
                            img_clip = (ImageClip(img_path)
                                        .set_start(rel_start)
                                        .set_duration(duration)
                                        .set_position("center")
                                        .resize(height=sub.h * 0.8)) # Resize to 80% of height
                            layers.append(img_clip)
                        except Exception as e:
                            print(f"Failed to create ImageClip: {e}")

            # --- Captions ---
            for cap in captions:
                c_start = float(cap.get("start", 0))
                c_end = float(cap.get("end", 0))
                text = cap.get("text", "")
                
                if not text:
                    continue
                
                # Check if caption starts in this segment
                if start <= c_start < end:
                    rel_start = c_start - start
                    # Cap end at segment end
                    actual_end = min(c_end, end)
                    rel_end = actual_end - start
                    duration = rel_end - rel_start
                    
                    if duration > 0.5: # Min duration
                        try:
                            # Using basic settings. Font might vary by system. 
                            # 'Amiri-Bold' or 'DejaVuSans' are often available on Linux.
                            # Passing font=None might fallback to default.
                            text_clip = (TextClip(text, fontsize=40, color='white', stroke_color='black', stroke_width=2, method='caption', size=(sub.w * 0.9, None))
                                         .set_start(rel_start)
                                         .set_duration(duration)
                                         .set_position(('center', 'bottom')))
                            layers.append(text_clip)
                        except Exception as e:
                            print(f"Failed to create TextClip. This is often due to ImageMagick configuration.")
                            print(f"Details: {e}")
                            print("Tip: Check if 'policy.xml' allows read/write for PDF/Text if on Linux.")

            if len(layers) > 1:
                combined = CompositeVideoClip(layers)
                clips.append(combined)
            else:
                clips.append(sub)

        # Concatenate
        if clips:
            print(f"Concatenating {len(clips)} clips...")
            try:
                final = concatenate_videoclips(clips, method="compose")
                # Write file
                final.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24)
                print(f"Video saved to {output_path}")
                return output_path
            except Exception as e:
                print(f"Error saving video: {e}")
                return None
        else:
            print("No clips generated.")
            return None
