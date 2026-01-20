from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, ImageClip, AudioFileClip, CompositeAudioClip
import moviepy.video.fx.all as vfx
import moviepy.audio.fx.all as afx
import moviepy.config as mp_config
import os
import bisect
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
        im_binary = os.environ.get("IMAGEMAGICK_BINARY")
        if im_binary:
            mp_config.change_settings({"IMAGEMAGICK_BINARY": im_binary})

    def edit(self, video_path: str, analysis_data: Dict[str, Any], graphic_paths: Dict[int, str], output_path: str = "output.mp4", options: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Edits the video based on the analysis data and generated graphics.

        Args:
            video_path (str): Path to the source video.
            analysis_data (Dict[str, Any]): Analysis results containing segments, captions, etc.
            graphic_paths (Dict[int, str]): A dictionary mapping graphic indices to file paths.
            output_path (str): Path to save the final video. Defaults to "output.mp4".
            options (Optional[Dict[str, Any]]): Additional editing options (music, effects, subtitles).

        Returns:
            Optional[str]: The path to the output video, or None if editing fails.
        """
        print(f"Editing video: {video_path}")
        if options is None:
            options = {}

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

        # --- Pre-process Graphics ---
        # Optimize graphics lookup by sorting and using bisect (O(M log M) + O(N log M)) instead of O(N * M)
        graphics_reqs = analysis_data.get("graphics", [])
        # Store as (timestamp, original_index, graphic_req)
        sorted_graphics = []
        for i, req in enumerate(graphics_reqs):
            t = float(req.get("timestamp", 0))
            sorted_graphics.append((t, i, req))

        # Sort by timestamp
        sorted_graphics.sort(key=lambda x: x[0])
        # Extract timestamps for bisect
        g_timestamps = [x[0] for x in sorted_graphics]
        
        graphics_reqs = analysis_data.get("graphics", [])
        captions = analysis_data.get("captions", [])

        # Subtitle settings
        sub_opts = options.get("subtitle", {})
        sub_font = sub_opts.get("font", "Arial")
        sub_fontsize = sub_opts.get("fontsize", 40)
        sub_color = sub_opts.get("color", "white")

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
            
            # Apply filters if any
            filter_type = options.get("filter")
            if filter_type == "bw":
                sub = vfx.blackwhite(sub)
            elif filter_type == "contrast":
                sub = vfx.lum_contrast(sub, lum=0, contrast=0.5)

            layers = [sub]
            
            # --- Graphics (Overlay) ---
            # Find relevant graphics using binary search
            idx_start = bisect.bisect_left(g_timestamps, start)
            idx_end = bisect.bisect_left(g_timestamps, end)

            for g_time, i, graphic_req in sorted_graphics[idx_start:idx_end]:
                # g_time is already guaranteed to be >= start and < end by bisect logic
                
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
                            # Using customizable settings
                            text_clip = (TextClip(text, fontsize=sub_fontsize, color=sub_color,
                                                  font=sub_font, stroke_color='black', stroke_width=2,
                                                  method='caption', size=(sub.w * 0.9, None))
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

            # Transitions (Crossfade)
            crossfade = options.get("crossfade", 0.0)
            padding = 0
            if crossfade > 0:
                print(f"Applying crossfade of {crossfade}s")
                # Apply crossfadein/out to create smooth transitions
                # We need to fade in all clips except the first, and we can also fade out all except last
                # But actually, standard way with padding is:
                # clip1 (fadeout), clip2 (fadein), overlap
                padding = -crossfade
                new_clips = []
                for i, clip in enumerate(clips):
                    if i > 0:
                        clip = clip.crossfadein(crossfade)
                        # also fade audio
                        if clip.audio:
                            clip = clip.set_audio(clip.audio.audio_fadein(crossfade))
                    if i < len(clips) - 1:
                        clip = clip.crossfadeout(crossfade)
                        if clip.audio:
                            clip = clip.set_audio(clip.audio.audio_fadeout(crossfade))
                    new_clips.append(clip)
                clips = new_clips

            try:
                final = concatenate_videoclips(clips, method="compose", padding=padding)

                # Background Music
                music_path = options.get("music_path")
                if music_path and os.path.exists(music_path):
                    print(f"Adding background music: {music_path}")
                    try:
                        music = AudioFileClip(music_path)
                        music_vol = options.get("music_volume", 0.5)

                        # Loop music if needed
                        if music.duration < final.duration:
                            music = afx.audio_loop(music, duration=final.duration)
                        else:
                            music = music.subclip(0, final.duration)

                        music = music.volumex(music_vol)

                        # Combine with original audio
                        original_audio = final.audio
                        if original_audio:
                            final_audio = CompositeAudioClip([original_audio, music])
                        else:
                            final_audio = music

                        final = final.set_audio(final_audio)
                    except Exception as e:
                        print(f"Failed to add background music: {e}")

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
