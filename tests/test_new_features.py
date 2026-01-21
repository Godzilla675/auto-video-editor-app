import sys
import unittest
from unittest.mock import MagicMock, patch
import importlib

# Mock dependencies before imports
# Ensure these mocks are consistent with other tests if running in the same process
mock_moviepy = MagicMock()
mock_moviepy_editor = MagicMock()
mock_vfx = MagicMock()
mock_afx = MagicMock()
mock_config = MagicMock()

sys.modules["moviepy"] = mock_moviepy
sys.modules["moviepy.editor"] = mock_moviepy_editor
sys.modules["moviepy.video.fx.all"] = mock_vfx
sys.modules["moviepy.audio.fx.all"] = mock_afx
sys.modules["moviepy.config"] = mock_config

import os
# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.editor
from src.editor import Editor

class TestNewFeatures(unittest.TestCase):

    def setUp(self):
        # Reload editor to pick up fresh mocks
        importlib.reload(src.editor)

        # Access the mocks via the imported module to ensure we are looking at the same objects
        self.mock_moviepy_editor = sys.modules["moviepy.editor"]
        self.mock_vfx = src.editor.vfx
        self.mock_afx = src.editor.afx

        # Setup standard mock returns
        self.mock_clip = MagicMock()
        self.mock_clip.duration = 10.0
        self.mock_clip.w = 100
        self.mock_clip.h = 100
        self.mock_clip.subclip.return_value = self.mock_clip
        # Need to return self for fluent interface calls like set_audio
        self.mock_clip.set_audio.return_value = self.mock_clip
        self.mock_clip.volumex.return_value = self.mock_clip
        self.mock_clip.set_duration.return_value = self.mock_clip

        self.mock_moviepy_editor.VideoFileClip.return_value = self.mock_clip
        self.mock_moviepy_editor.TextClip.return_value = MagicMock()
        self.mock_moviepy_editor.ImageClip.return_value = MagicMock()
        self.mock_moviepy_editor.CompositeVideoClip.return_value = MagicMock()

        # concatenate returns a new clip
        self.mock_final = MagicMock()
        self.mock_final.duration = 10.0
        self.mock_final.set_audio.return_value = self.mock_final
        self.mock_moviepy_editor.concatenate_videoclips.return_value = self.mock_final

        # AudioFileClip
        self.mock_audio = MagicMock()
        self.mock_audio.duration = 5.0
        self.mock_audio.volumex.return_value = self.mock_audio
        self.mock_audio.set_duration.return_value = self.mock_audio
        self.mock_moviepy_editor.AudioFileClip.return_value = self.mock_audio

    def test_background_music(self):
        print("Testing Background Music...")
        editor = src.editor.Editor()

        # Mock looping
        self.mock_afx.audio_loop.return_value = self.mock_audio

        analysis_data = {"segments": [{"start": 0, "end": 10}]}

        with patch('os.path.exists', return_value=True):
            editor.edit("video.mp4", analysis_data, {}, music_path="music.mp3", music_volume=0.5)

            # Verify AudioFileClip was loaded
            self.mock_moviepy_editor.AudioFileClip.assert_called_with("music.mp3")
            # Verify volume set
            self.mock_audio.volumex.assert_called_with(0.5)
            # Verify loop called (since music duration 5.0 < video 10.0)
            self.mock_afx.audio_loop.assert_called()
            # Verify audio set on final clip
            self.mock_final.set_audio.assert_called()

    def test_crossfade(self):
        print("Testing Crossfade...")
        editor = src.editor.Editor()

        analysis_data = {
            "segments": [
                {"start": 0, "end": 5},
                {"start": 5, "end": 10}
            ]
        }

        # Mock vfx
        self.mock_vfx.crossfadein.return_value = self.mock_clip
        self.mock_vfx.crossfadeout.return_value = self.mock_clip

        with patch('os.path.exists', return_value=True):
            editor.edit("video.mp4", analysis_data, {}, crossfade=1.0)

            # Verify crossfade fx applied
            self.mock_vfx.crossfadein.assert_called()
            self.mock_vfx.crossfadeout.assert_called()

            # Verify concatenate called with padding
            # call_args is (args, kwargs)
            args, kwargs = self.mock_moviepy_editor.concatenate_videoclips.call_args
            self.assertEqual(kwargs.get("padding"), -1.0)

    def test_visual_filter(self):
        print("Testing Visual Filter...")
        editor = src.editor.Editor()

        analysis_data = {"segments": [{"start": 0, "end": 5}]}

        self.mock_vfx.blackwhite.return_value = self.mock_clip

        with patch('os.path.exists', return_value=True):
            editor.edit("video.mp4", analysis_data, {}, visual_filter="bw")

            self.mock_vfx.blackwhite.assert_called()

    def test_subtitle_styling(self):
        print("Testing Subtitle Styling...")
        editor = src.editor.Editor()

        analysis_data = {
            "segments": [{"start": 0, "end": 5}],
            "captions": [{"start": 1, "end": 4, "text": "Subtitle"}]
        }

        style = {
            "font": "MyFont",
            "fontsize": 50,
            "color": "red",
            "stroke_color": "blue",
            "stroke_width": 3
        }

        with patch('os.path.exists', return_value=True):
            editor.edit("video.mp4", analysis_data, {}, subtitle_style=style)

            # Verify TextClip called with correct args
            args, kwargs = self.mock_moviepy_editor.TextClip.call_args
            self.assertEqual(args[0], "Subtitle")
            self.assertEqual(kwargs["font"], "MyFont")
            self.assertEqual(kwargs["fontsize"], 50)
            self.assertEqual(kwargs["color"], "red")
            self.assertEqual(kwargs["stroke_color"], "blue")
            self.assertEqual(kwargs["stroke_width"], 3)

if __name__ == '__main__':
    unittest.main()
