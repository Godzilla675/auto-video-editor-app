from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, ImageClip, AudioFileClip, CompositeAudioClip, afx
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

    def edit(self,
             video_path: str,
             analysis_data: Dict[str, Any],
             graphic_paths: Dict[int, str],
             output_path: str = "output.mp4",
             music_path: Optional[str] = None,
             music_volume: float = 0.1,
             intro_path: Optional[str] = None,
             outro_path: Optional[str] = None,
             subtitle_config: Optional[Dict[str, Any]] = None
             ) -> Optional[str]:
        """
        Edits the video based on the analysis data and generated graphics.

        Args:
            video_path (str): Path to the source video.
            analysis_data (Dict[str, Any]): Analysis results containing segments, captions, etc.
            graphic_paths (Dict[int, str]): A dictionary mapping graphic indices to file paths.
            output_path (str): Path to save the final video. Defaults to "output.mp4".
            music_path (Optional[str]): Path to background music.
            music_volume (float): Volume of background music.
            intro_path (Optional[str]): Path to intro video.
            outro_path (Optional[str]): Path to outro video.
            subtitle_config (Optional[Dict[str, Any]]): Config for subtitle styling.

        Returns:
            Optional[str]: The path to the output video, or None if editing fails.
        """
        if subtitle_config is None:
            subtitle_config = {}

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
        
        captions = analysis_data.get("captions", [])

        main_clips = []

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
                            # Use styles from subtitle_config
                            fontsize = subtitle_config.get("fontsize", 40)
                            font = subtitle_config.get("font", "Arial")
                            color = subtitle_config.get("color", "white")
                            stroke_color = subtitle_config.get("stroke_color", "black")
                            stroke_width = subtitle_config.get("stroke_width", 1.5)

                            text_clip = (TextClip(text, fontsize=fontsize, font=font, color=color,
                                                 stroke_color=stroke_color, stroke_width=stroke_width,
                                                 method='caption', size=(sub.w * 0.9, None))
                                         .set_start(rel_start)
                                         .set_duration(duration)
                                         .set_position(('center', 'bottom')))
                            layers.append(text_clip)
                        except Exception as e:
                            print(f"Failed to create TextClip. This is often due to ImageMagick configuration.")
                            print(f"Details: {e}")

            if len(layers) > 1:
                combined = CompositeVideoClip(layers)
                main_clips.append(combined)
            else:
                main_clips.append(sub)

        # Handle Transitions (Crossfade)
        # Check if analysis_data has transitions
        transitions = analysis_data.get("transitions", [])
        # We can implement basic crossfade between all main clips if requested
        # Or parse the transitions list. For now, let's just do crossfades if we have multiple clips
        # and transitions are suggested.
        # But simple crossfade:

        # A more robust approach for transitions based on timestamp is complex because we just cut segments.
        # If the user wants crossfades, we typically apply them between segments.

        final_clips = []
        if main_clips:
            # Check if any transition is type "crossfade"
            should_crossfade = any(t.get("type") == "crossfade" for t in transitions)

            if should_crossfade and len(main_clips) > 1:
                print("Applying crossfade transitions...")
                padding = 1.0 # 1 second overlap
                # We need to adjust start times or use composite video clip with overlapping start times
                # concatenate_videoclips supports 'padding' with method='compose' but it simply places them one after another with negative padding
                # To do real crossfade, we need to add fadein/fadeout effects

                processed_clips = []
                for i, clip in enumerate(main_clips):
                    c = clip
                    if i > 0:
                        c = c.crossfadein(padding)
                    processed_clips.append(c)

                # Join with padding (negative padding creates overlap)
                try:
                    final_main = concatenate_videoclips(processed_clips, method="compose", padding=-padding)
                except Exception as e:
                    print(f"Crossfade failed: {e}. Falling back to simple concatenation.")
                    final_main = concatenate_videoclips(main_clips, method="compose")
            else:
                final_main = concatenate_videoclips(main_clips, method="compose")
        else:
            print("No clips generated.")
            return None

        # --- Intro & Outro ---
        sequence = []

        if intro_path and os.path.exists(intro_path):
            print(f"Adding intro: {intro_path}")
            try:
                intro_clip = VideoFileClip(intro_path)
                # Resize intro to match main video if needed?
                # Ideally videos should match resolution. We'll attempt resize.
                if intro_clip.size != final_main.size:
                     intro_clip = intro_clip.resize(final_main.size)
                sequence.append(intro_clip)
            except Exception as e:
                print(f"Failed to load intro: {e}")

        sequence.append(final_main)

        if outro_path and os.path.exists(outro_path):
             print(f"Adding outro: {outro_path}")
             try:
                outro_clip = VideoFileClip(outro_path)
                if outro_clip.size != final_main.size:
                     outro_clip = outro_clip.resize(final_main.size)
                sequence.append(outro_clip)
             except Exception as e:
                print(f"Failed to load outro: {e}")

        if len(sequence) > 1:
            final_video = concatenate_videoclips(sequence, method="compose")
        else:
            final_video = final_main

        # --- Background Music ---
        if music_path and os.path.exists(music_path):
            print(f"Adding background music: {music_path}")
            try:
                bg_music = AudioFileClip(music_path)

                # Loop music if shorter than video
                if bg_music.duration < final_video.duration:
                    bg_music = afx.audio_loop(bg_music, duration=final_video.duration)
                else:
                    bg_music = bg_music.subclip(0, final_video.duration)

                # Set volume
                bg_music = bg_music.volumex(music_volume)

                # Mix with original audio
                original_audio = final_video.audio
                if original_audio:
                    mixed_audio = CompositeAudioClip([original_audio, bg_music])
                else:
                    mixed_audio = bg_music

                final_video = final_video.set_audio(mixed_audio)

            except Exception as e:
                print(f"Failed to add background music: {e}")

        # Write file
        print(f"Saving to {output_path}...")
        try:
            final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24)
            print(f"Video saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving video: {e}")
            return None
