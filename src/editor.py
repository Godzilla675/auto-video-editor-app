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

    def _create_title_card(self, text: str, duration: float = 3.0, size: tuple = (1920, 1080)) -> CompositeVideoClip:
        """
        Creates a simple title card with text on a black background.
        """
        bg = ColorClip(size, color=(0, 0, 0), duration=duration)
        txt_clip = (TextClip(text, fontsize=70, color='white', font='DejaVuSans', size=size, method='caption')
                    .set_position('center')
                    .set_duration(duration))
        return CompositeVideoClip([bg, txt_clip])

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
            "stroke_width": 2,
            "box_color": "black",
            "box_opacity": 0.5
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
                                # Apply Ken Burns effect (zoom)
                                # Start at 80% height, zoom in by 4% per second
                                img_clip = (ImageClip(img_path)
                                            .set_start(rel_start)
                                            .set_duration(duration)
                                            .set_position("center")
                                            .resize(height=sub.h * 0.8)
                                            .resize(lambda t: 1 + 0.04 * t))
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
                            # Create TextClip
                            text_clip = (TextClip(text,
                                                  fontsize=sub_conf["fontsize"],
                                                  color=sub_conf["color"],
                                                  stroke_color=sub_conf["stroke_color"],
                                                  stroke_width=sub_conf["stroke_width"],
                                                  font=sub_conf["font"],
                                                  method='caption',
                                                  size=(sub.w * 0.9, None))
                                         .set_position(('center', 'bottom')))

                            # Create background box if requested
                            if sub_conf.get("box_opacity", 0) > 0:
                                # Get size of text clip
                                w, h = text_clip.size
                                # Add padding
                                box = ColorClip(size=(w + 20, h + 10), color=sub_conf.get("box_color", "black"))
                                box = box.set_opacity(sub_conf["box_opacity"]).set_position(('center', 'bottom'))

                                # Composite text on box
                                subtitle_composite = CompositeVideoClip([box, text_clip], use_bgclip=False)
                                subtitle_composite = (subtitle_composite
                                                      .set_start(rel_start)
                                                      .set_duration(duration)
                                                      .set_position(('center', 'bottom')))
                                layers.append(subtitle_composite)
                            else:
                                text_clip = text_clip.set_start(rel_start).set_duration(duration)
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
            # Add Intro
            if intro_text:
                print(f"Adding intro: {intro_text}")
                # Use size of the first clip to match dimensions
                size = (clips[0].w, clips[0].h)
                intro_clip = self._create_title_card(intro_text, size=size)
                clips.insert(0, intro_clip)

            # Add Outro
            if outro_text:
                print(f"Adding outro: {outro_text}")
                # Use size of the last clip (before insertion)
                size = (clips[0].w, clips[0].h)
                outro_clip = self._create_title_card(outro_text, size=size)
                clips.append(outro_clip)

            print(f"Concatenating {len(clips)} clips...")

            # Apply crossfade if requested
            padding = 0
            if crossfade > 0 and len(clips) > 1:
                print(f"Applying crossfade of {crossfade}s")
                # Apply crossfadein to all clips except the first one
                clips = [clip.crossfadein(crossfade) if i > 0 else clip for i, clip in enumerate(clips)]
                padding = -crossfade

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
                    elif visual_filter == 'rotate_90':
                         final = vfx.rotate(final, angle=90)
                    elif visual_filter == 'rotate_180':
                         final = vfx.rotate(final, angle=180)
                    elif visual_filter == 'mirror_x':
                         final = vfx.mirror_x(final)
                    elif visual_filter == 'mirror_y':
                         final = vfx.mirror_y(final)

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

                        # Apply Fade In/Out
                        music_clip = afx.audio_fadein(music_clip, 2.0)
                        music_clip = afx.audio_fadeout(music_clip, 2.0)

                        # Mix
                        final_audio = CompositeAudioClip([final.audio, music_clip])
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
