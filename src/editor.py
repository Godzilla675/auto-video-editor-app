from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, ImageClip, AudioFileClip, CompositeAudioClip
import moviepy.video.fx.all as vfx
import moviepy.audio.fx.all as afx
import moviepy.config as mp_config
import os
import bisect
from typing import Dict, Any, List, Optional, Union

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
             background_music: Optional[str] = None,
             music_volume: float = 0.1,
             crossfade_duration: float = 0.0,
             visual_filter: Optional[str] = None,
             subtitle_config: Optional[Dict[str, Any]] = None
             ) -> Optional[str]:
        """
        Edits the video based on the analysis data and generated graphics.

        Args:
            video_path (str): Path to the source video.
            analysis_data (Dict[str, Any]): Analysis results containing segments, captions, etc.
            graphic_paths (Dict[int, str]): A dictionary mapping graphic indices to file paths.
            output_path (str): Path to save the final video. Defaults to "output.mp4".
            background_music (Optional[str]): Path to background music file.
            music_volume (float): Volume of background music (0.0 to 1.0).
            crossfade_duration (float): Duration of crossfade transition in seconds.
            visual_filter (Optional[str]): Visual filter to apply (e.g., 'black_white').
            subtitle_config (Optional[Dict[str, Any]]): Configuration for subtitles (font, size, etc.).

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

        # Subtitle defaults
        if subtitle_config is None:
            subtitle_config = {}
        sub_font = subtitle_config.get("font")
        sub_fontsize = subtitle_config.get("fontsize", 40)
        sub_color = subtitle_config.get("color", 'white')
        sub_stroke_color = subtitle_config.get("stroke_color", 'black')
        sub_stroke_width = subtitle_config.get("stroke_width", 2)

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
            
            # Apply visual filter
            if visual_filter == 'black_white':
                sub = sub.fx(vfx.blackwhite)
            elif visual_filter:
                print(f"Warning: Unknown filter '{visual_filter}' ignored.")

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
                            # Using configured settings
                            text_clip = (TextClip(text,
                                                 font=sub_font,
                                                 fontsize=sub_fontsize,
                                                 color=sub_color,
                                                 stroke_color=sub_stroke_color,
                                                 stroke_width=sub_stroke_width,
                                                 method='caption',
                                                 size=(sub.w * 0.9, None))
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

        # Apply crossfade if requested and possible
        if crossfade_duration > 0 and len(clips) > 1:
            print(f"Applying crossfade of {crossfade_duration}s")
            # To crossfade, clips must overlap.
            # padding = -crossfade_duration
            # clip1.crossfadein(d) -> starts black and fades in? No.
            # MoviePy compositing/concatenation handles crossfades if set up correctly.
            # Standard way: clip.crossfadein(d) makes it fade in from black/transparent.
            # clip.crossfadeout(d) makes it fade out to black/transparent.
            # But for transitions between two clips A and B:
            # We want A to fade out while B fades in (dissolve).
            # concatenate_videoclips(clips, method="compose") can handle overlapping if padding is negative.

            # Correct approach for crossfade transition in MoviePy v1:
            # Set crossfadein on the second clip (and subsequent).
            # And use padding equal to negative duration.

            processed_clips = []
            for i, clip in enumerate(clips):
                if i > 0:
                     # Fade in this clip
                     clip = clip.crossfadein(crossfade_duration)
                if i < len(clips) - 1:
                    # Fade out previous clip?
                    # crossfadein on B combined with overlap handles the transition A->B
                    pass

                # Set audio fade as well for smooth audio transition
                if clip.audio:
                    if i > 0:
                        clip = clip.set_audio(clip.audio.audio_fadein(crossfade_duration))
                    if i < len(clips) - 1:
                        clip = clip.set_audio(clip.audio.audio_fadeout(crossfade_duration))

                processed_clips.append(clip)

            try:
                final = concatenate_videoclips(processed_clips, method="compose", padding=-crossfade_duration)
            except Exception as e:
                print(f"Error applying crossfade: {e}. Falling back to standard concatenation.")
                final = concatenate_videoclips(clips, method="compose")
        elif clips:
            print(f"Concatenating {len(clips)} clips...")
            final = concatenate_videoclips(clips, method="compose")
        else:
            print("No clips generated.")
            return None

        # Add background music
        if background_music and os.path.exists(background_music):
            print(f"Adding background music: {background_music}")
            try:
                bg_music = AudioFileClip(background_music)

                # Loop music if video is longer
                if bg_music.duration < final.duration:
                    bg_music = afx.audio_loop(bg_music, duration=final.duration)
                else:
                    bg_music = bg_music.subclip(0, final.duration)

                bg_music = bg_music.volumex(music_volume)

                # Composite audio
                if final.audio:
                    final_audio = CompositeAudioClip([final.audio, bg_music])
                else:
                    final_audio = bg_music

                final = final.set_audio(final_audio)

            except Exception as e:
                print(f"Error adding background music: {e}")

        # Write file
        try:
            final.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24)
            print(f"Video saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving video: {e}")
            return None
