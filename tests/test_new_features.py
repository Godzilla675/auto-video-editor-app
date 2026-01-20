import sys
import unittest
from unittest.mock import MagicMock, patch
import os

# Mock dependencies before import
mock_moviepy = MagicMock()
mock_editor = MagicMock()
mock_vfx = MagicMock()
mock_afx = MagicMock()
mock_config = MagicMock()

sys.modules["moviepy"] = mock_moviepy
sys.modules["moviepy.editor"] = mock_editor
sys.modules["moviepy.video.fx.all"] = mock_vfx
sys.modules["moviepy.audio.fx.all"] = mock_afx
sys.modules["moviepy.config"] = mock_config

sys.modules["whisper"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["numpy"] = MagicMock()

# Import Editor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import importlib
import src.editor

class TestNewFeatures(unittest.TestCase):

    def setUp(self):
        # We must reload src.editor to ensure it captures the current mocks in sys.modules
        importlib.reload(src.editor)
        from src.editor import Editor
        self.Editor = Editor

        # Reset mocks
        mock_editor.reset_mock()
        mock_vfx.reset_mock()
        mock_afx.reset_mock()

        # Setup standard mock returns
        self.mock_clip = MagicMock()
        self.mock_clip.duration = 10.0
        self.mock_clip.w = 100
        self.mock_clip.h = 100
        self.mock_clip.subclip.return_value = self.mock_clip
        # Fluent interface needs to return self/mock
        self.mock_clip.resize.return_value = self.mock_clip
        self.mock_clip.set_start.return_value = self.mock_clip
        self.mock_clip.set_duration.return_value = self.mock_clip
        self.mock_clip.set_position.return_value = self.mock_clip
        self.mock_clip.set_audio.return_value = self.mock_clip
        self.mock_clip.crossfadein.return_value = self.mock_clip
        self.mock_clip.crossfadeout.return_value = self.mock_clip
        self.mock_clip.audio.audio_fadein.return_value = MagicMock()
        self.mock_clip.audio.audio_fadeout.return_value = MagicMock()

        mock_editor.VideoFileClip.return_value = self.mock_clip
        mock_editor.AudioFileClip.return_value = MagicMock()
        mock_editor.TextClip.return_value = self.mock_clip
        mock_editor.ImageClip.return_value = self.mock_clip
        mock_editor.CompositeVideoClip.return_value = self.mock_clip

        self.mock_final = MagicMock()
        self.mock_final.duration = 20.0 # concatenated duration
        self.mock_final.set_audio.return_value = self.mock_final
        mock_editor.concatenate_videoclips.return_value = self.mock_final

    def test_music_integration(self):
        print("Testing Music Integration...")
        editor = self.Editor()
        analysis_data = {"segments": [{"start": 0, "end": 5}]}
        graphic_paths = {}
        options = {
            "music_path": "music.mp3",
            "music_volume": 0.3
        }

        # Setup music mock with duration to avoid comparison error
        mock_music = MagicMock()
        mock_music.duration = 10.0 # shorter than final (20.0), so loop should be called
        mock_music.volumex.return_value = mock_music
        mock_music.subclip.return_value = mock_music
        mock_editor.AudioFileClip.return_value = mock_music

        # Ensure we configure the afx that src.editor is using
        src.editor.afx.audio_loop.return_value = mock_music

        with patch('os.path.exists', return_value=True):
            editor.edit("video.mp4", analysis_data, graphic_paths, options=options)

        # Verify AudioFileClip loaded
        mock_editor.AudioFileClip.assert_called_with("music.mp3")

        # Verify loop called because music.duration (10) < final.duration (20)
        src.editor.afx.audio_loop.assert_called()

        # Verify volumex called
        mock_music.volumex.assert_called_with(0.3)

        # Verify set_audio called on final clip
        self.mock_final.set_audio.assert_called()

    def test_filters(self):
        print("Testing Filters...")
        editor = self.Editor()
        analysis_data = {"segments": [{"start": 0, "end": 5}]}
        graphic_paths = {}

        # Test BW
        options = {"filter": "bw"}
        with patch('os.path.exists', return_value=True):
            editor.edit("video.mp4", analysis_data, graphic_paths, options=options)
        src.editor.vfx.blackwhite.assert_called()

        # Test Contrast
        src.editor.vfx.reset_mock()
        options = {"filter": "contrast"}
        with patch('os.path.exists', return_value=True):
            editor.edit("video.mp4", analysis_data, graphic_paths, options=options)
        src.editor.vfx.lum_contrast.assert_called()

    def test_transitions(self):
        print("Testing Transitions...")
        editor = self.Editor()
        # Need at least 2 segments for transition logic to be apparent
        analysis_data = {"segments": [{"start": 0, "end": 5}, {"start": 5, "end": 10}]}
        graphic_paths = {}
        options = {"crossfade": 1.0}

        with patch('os.path.exists', return_value=True):
            editor.edit("video.mp4", analysis_data, graphic_paths, options=options)

        # Verify crossfadein/out called on clips
        # Note: mocks return self, so call count accumulates
        self.assertTrue(self.mock_clip.crossfadein.called or self.mock_clip.crossfadeout.called)

        # Verify concatenate called with padding
        mock_editor.concatenate_videoclips.assert_called()
        args, kwargs = mock_editor.concatenate_videoclips.call_args
        self.assertEqual(kwargs.get("padding"), -1.0)

    def test_subtitle_customization(self):
        print("Testing Subtitle Customization...")
        editor = self.Editor()
        analysis_data = {
            "segments": [{"start": 0, "end": 5}],
            "captions": [{"start": 1, "end": 2, "text": "Hi"}]
        }
        graphic_paths = {}
        options = {
            "subtitle": {
                "font": "MyFont",
                "fontsize": 50,
                "color": "red"
            }
        }

        with patch('os.path.exists', return_value=True):
            editor.edit("video.mp4", analysis_data, graphic_paths, options=options)

        mock_editor.TextClip.assert_called()
        args, kwargs = mock_editor.TextClip.call_args
        self.assertEqual(kwargs.get("font"), "MyFont")
        self.assertEqual(kwargs.get("fontsize"), 50)
        self.assertEqual(kwargs.get("color"), "red")

if __name__ == '__main__':
    unittest.main()
