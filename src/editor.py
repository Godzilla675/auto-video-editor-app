from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, ImageClip, AudioFileClip, CompositeAudioClip, ColorClip
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

    def _create_title_card(self, text: str, duration: float, size: tuple) -> Optional[CompositeVideoClip]:
        try:
             # Create black background
            bg = ColorClip(size=size, color=(0,0,0), duration=duration)

            # Create text
            txt = (TextClip(text, fontsize=70, color='white', font='DejaVuSans', size=size, method='caption')
                   .set_position('center')
                   .set_duration(duration))

            return CompositeVideoClip([bg, txt]).set_duration(duration)
        except Exception as e:
            print(f"Error creating title card: {e}")
            return None

    def edit(self, video_path: str, analysis_data: Dict[str, Any], graphic_paths: Dict[int, str], output_path: str = "output.mp4",
             music: Optional[str] = None, music_volume: float = 0.1, crossfade: float = 0.0,
             subtitle_config: Optional[Dict[str, Any]] = None, visual_filter: Optional[str] = None,
             intro_text: Optional[str] = None, outro_text: Optional[str] = None) -> Optional[str]:
        """
        Edits the video based on the analysis data and generated graphics.

        Args:
            video_path (str): Path to the source video.
            analysis_data (Dict[str, Any]): Analysis results containing segments, captions, etc.
            graphic_paths (Dict[int, str]): A dictionary mapping graphic indices to file paths.
            output_path (str): Path to save the final video. Defaults to "output.mp4".
            music (Optional[str]): Path to background music file.
            music_volume (float): Volume of background music (0.0 to 1.0).
            crossfade (float): Duration of crossfade transition between clips.
            subtitle_config (Optional[Dict[str, Any]]): Configuration for subtitles (font, size, color, etc.).
            visual_filter (Optional[str]): Name of visual filter to apply (e.g., 'black_white').
            intro_text (Optional[str]): Text for the intro title card.
            outro_text (Optional[str]): Text for the outro title card.

        Returns:
            Optional[str]: The path to the output video, or None if editing fails.
        """
        print(f"Editing video: {video_path}")

        # Defaults for subtitle config
        sub_conf = {
            "font": "DejaVuSans",
            "fontsize": 40,
            "color": "white",
            "stroke_color": "black",
            "stroke_width": 2
        }
        if subtitle_config:
            sub_conf.update(subtitle_config)

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
            # Find relevant graphics. We need graphics that OVERLAP with [start, end).
            # Overlap condition: graphic_start < end AND graphic_end > start.

            # Since sorted_graphics is sorted by start time, we can stop checking when graphic_start >= end.
            idx_end = bisect.bisect_left(g_timestamps, end)

            for g_time, i, graphic_req in sorted_graphics[:idx_end]:
                img_path = graphic_paths.get(i)
                if img_path and os.path.exists(img_path):
                    g_duration = float(graphic_req.get("duration", 3.0))
                    g_end = g_time + g_duration

                    if g_end > start:
                        # It overlaps!
                        # Calculate relative start and duration for this segment.
                        rel_start = max(0, g_time - start)

                        overlap_start = max(g_time, start)
                        overlap_end = min(g_end, end)
                        duration = overlap_end - overlap_start

                        if duration > 0:
                            print(f"Adding graphic {img_path} at relative {rel_start}s with duration {duration}s")
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
                
                # Check for overlap: caption ends after segment starts AND caption starts before segment ends
                if c_end > start and c_start < end:
                    rel_start = max(0, c_start - start)

                    overlap_start = max(c_start, start)
                    overlap_end = min(c_end, end)
                    duration = overlap_end - overlap_start
                    
                    if duration > 0.5: # Min duration
                        try:
                            # Using basic settings. Font might vary by system. 
                            # 'Amiri-Bold' or 'DejaVuSans' are often available on Linux.
                            # Passing font=None might fallback to default.
                            text_clip = (TextClip(text,
                                                  fontsize=sub_conf["fontsize"],
                                                  color=sub_conf["color"],
                                                  stroke_color=sub_conf["stroke_color"],
                                                  stroke_width=sub_conf["stroke_width"],
                                                  font=sub_conf["font"],
                                                  method='caption',
                                                  size=(sub.w * 0.9, None))
                                         .set_start(rel_start)
                                         .set_duration(duration)
                                         .set_position(('center', 'bottom')))

                            if sub_conf.get("box_color"):
                                box_color = sub_conf.get("box_color")
                                # Map color name to value if needed, or rely on moviepy string colors
                                # box_opacity = float(sub_conf.get("box_opacity", 0.6))
                                # MoviePy ColorClip takes color argument.

                                # Note: TextClip text is centered in the box size.
                                # To do a background box properly, we need the size of the text.
                                # But method='caption' forces size.

                                # A simple way is to make a ColorClip of the same size as TextClip
                                # But TextClip size is (sub.w * 0.9, None).
                                # Getting actual text size is hard without rendering.

                                # Let's try creating a background strip at the bottom.
                                bg_h = sub_conf["fontsize"] * 1.5 # Estimate height
                                bg_clip = (ColorClip(size=(int(sub.w), int(bg_h)), color=box_color)
                                           .set_opacity(float(sub_conf.get("box_opacity", 0.6)))
                                           .set_start(0) # Relative to composite
                                           .set_duration(duration)
                                           .set_position(('center', 'bottom')))

                                # Reset text start for composite
                                text_clip = text_clip.set_start(0).set_position(('center', 'bottom'))

                                # Combine
                                composite_sub = CompositeVideoClip([bg_clip, text_clip], size=sub.size).set_start(rel_start).set_duration(duration).set_position(('center', 'bottom'))
                                layers.append(composite_sub)
                            else:
                                layers.append(text_clip)

                        except Exception as e:
                            print(f"Failed to create TextClip. This is often due to ImageMagick configuration.")
                            print(f"Details: {e}")
                            print("Tip: Check if 'policy.xml' allows read/write for PDF/Text if on Linux.")

            if len(layers) > 1:
                # IMPORTANT: CompositeVideoClip duration must be explicitly set to the base clip's duration
                combined = CompositeVideoClip(layers).set_duration(sub.duration)
                clips.append(combined)
            else:
                clips.append(sub)

        # Concatenate
        if clips:

            # Intro
            if intro_text:
                print(f"Adding intro: {intro_text}")
                intro_clip = self._create_title_card(intro_text, 3.0, video.size)
                if intro_clip:
                    clips.insert(0, intro_clip)

            # Outro
            if outro_text:
                print(f"Adding outro: {outro_text}")
                outro_clip = self._create_title_card(outro_text, 3.0, video.size)
                if outro_clip:
                    clips.append(outro_clip)

            print(f"Concatenating {len(clips)} clips...")

            # Apply crossfade if requested
            padding = 0
            if crossfade > 0 and len(clips) > 1:
                # Dynamic capping of crossfade
                min_duration = min(c.duration for c in clips)
                actual_crossfade = min(crossfade, max(0, min_duration - 0.1))

                print(f"Applying crossfade of {actual_crossfade}s")
                # Apply crossfadein to all clips except the first one
                clips = [clip.crossfadein(actual_crossfade) if i > 0 else clip for i, clip in enumerate(clips)]
                padding = -actual_crossfade

            try:
                final = concatenate_videoclips(clips, method="compose", padding=padding)

                # Apply Visual Filter
                if visual_filter:
                    print(f"Applying visual filter: {visual_filter}")
                    if visual_filter == 'black_white':
                        final = vfx.blackwhite(final)
                    elif visual_filter == 'invert_colors':
                        final = vfx.invert_colors(final)
                    elif visual_filter == 'painting':
                        final = vfx.painting(final)
                    # Add more filters as needed

                # Apply Background Music
                if music and os.path.exists(music):
                    print(f"Adding background music: {music}")
                    try:
                        music_clip = AudioFileClip(music)

                        # Loop if shorter
                        if music_clip.duration < final.duration:
                            music_clip = afx.audio_loop(music_clip, duration=final.duration)
                        else:
                            music_clip = music_clip.subclip(0, final.duration)

                        # Adjust volume
                        music_clip = music_clip.volumex(music_volume)

                        # Mix
                        if final.audio is not None:
                            final_audio = CompositeAudioClip([final.audio, music_clip])
                        else:
                            final_audio = music_clip

                        final = final.set_audio(final_audio)
                    except Exception as e:
                        print(f"Error adding background music: {e}")

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
