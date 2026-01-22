import sys
from unittest.mock import MagicMock, patch
import unittest
import os
import importlib

# Setup global mocks for new features if they don't exist
if "moviepy" not in sys.modules:
    sys.modules["moviepy"] = MagicMock()
if "moviepy.editor" not in sys.modules:
    sys.modules["moviepy.editor"] = MagicMock()
if "moviepy.config" not in sys.modules:
    sys.modules["moviepy.config"] = MagicMock()
if "moviepy.video.fx.all" not in sys.modules:
    sys.modules["moviepy.video.fx.all"] = MagicMock()
if "moviepy.audio.fx.all" not in sys.modules:
    sys.modules["moviepy.audio.fx.all"] = MagicMock()

# Ensure other dependencies are mocked
if "whisper" not in sys.modules: sys.modules["whisper"] = MagicMock()
if "openai" not in sys.modules: sys.modules["openai"] = MagicMock()
if "cv2" not in sys.modules: sys.modules["cv2"] = MagicMock()
if "PIL" not in sys.modules: sys.modules["PIL"] = MagicMock()
if "requests" not in sys.modules: sys.modules["requests"] = MagicMock()
if "numpy" not in sys.modules: sys.modules["numpy"] = MagicMock()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.editor

class TestNewFeatures(unittest.TestCase):
    def setUp(self):
        self.mock_vfx = MagicMock()
        sys.modules["moviepy.video.fx.all"] = self.mock_vfx

        self.mock_afx = MagicMock()
        sys.modules["moviepy.audio.fx.all"] = self.mock_afx

        importlib.reload(src.editor)

        # Force the module attributes to match our mocks, avoiding import system discrepancies
        src.editor.vfx = self.mock_vfx
        src.editor.afx = self.mock_afx

        self.mock_editor = sys.modules["moviepy.editor"]

        # Setup common mock returns
        self.mock_clip = MagicMock()
        self.mock_clip.duration = 10.0
        self.mock_clip.w = 100
        self.mock_clip.h = 100
        # Fluent interface returns the mock itself
        self.mock_clip.subclip.return_value = self.mock_clip
        self.mock_clip.fx.return_value = self.mock_clip
        self.mock_clip.resize.return_value = self.mock_clip
        self.mock_clip.set_start.return_value = self.mock_clip
        self.mock_clip.set_duration.return_value = self.mock_clip
        self.mock_clip.set_position.return_value = self.mock_clip
        self.mock_clip.volumex.return_value = self.mock_clip
        self.mock_clip.set_audio.return_value = self.mock_clip
        self.mock_clip.audio = MagicMock()

        self.mock_editor.VideoFileClip.return_value = self.mock_clip
        self.mock_editor.AudioFileClip.return_value = self.mock_clip
        self.mock_editor.CompositeVideoClip.return_value = self.mock_clip
        self.mock_editor.CompositeAudioClip.return_value = self.mock_clip
        self.mock_editor.concatenate_videoclips.return_value = self.mock_clip
        self.mock_editor.TextClip.return_value = self.mock_clip
        self.mock_editor.ImageClip.return_value = self.mock_clip

        self.mock_afx.audio_loop.return_value = self.mock_clip

        self.editor = src.editor.Editor()
        self.analysis_data = {"segments": [{"start": 0, "end": 5}]}
        self.graphic_paths = {}

    def test_music_integration(self):
        print("Testing Background Music...")
        with patch('os.path.exists', return_value=True):
            output = self.editor.edit(
                "dummy.mp4",
                self.analysis_data,
                self.graphic_paths,
                music_path="music.mp3",
                music_volume=0.5
            )

            self.mock_editor.AudioFileClip.assert_called_with("music.mp3")
            self.mock_clip.volumex.assert_called_with(0.5)
            self.mock_editor.CompositeAudioClip.assert_called()
            self.mock_clip.set_audio.assert_called()

    def test_filter_bw(self):
        print("Testing BW Filter...")
        with patch('os.path.exists', return_value=True):
            self.editor.edit(
                "dummy.mp4",
                self.analysis_data,
                self.graphic_paths,
                filter_type="bw"
            )
            self.mock_vfx.blackwhite.assert_called()

    def test_crossfade(self):
        print("Testing Crossfade...")
        with patch('os.path.exists', return_value=True):
            self.editor.edit(
                "dummy.mp4",
                self.analysis_data,
                self.graphic_paths,
                crossfade=1.0
            )

            # Check padding in concatenate
            self.mock_editor.concatenate_videoclips.assert_called()
            args, kwargs = self.mock_editor.concatenate_videoclips.call_args
            self.assertEqual(kwargs.get("padding"), -1.0)

            # Check if fadein/fadeout were applied
            self.mock_clip.fx.assert_any_call(self.mock_vfx.fadein, 1.0)
            self.mock_clip.fx.assert_any_call(self.mock_vfx.fadeout, 1.0)

    def test_subtitle_config(self):
        print("Testing Subtitle Config...")
        data = {
            "segments": [{"start": 0, "end": 5}],
            "captions": [{"start": 1, "end": 2, "text": "Sub"}]
        }

        with patch('os.path.exists', return_value=True):
            self.editor.edit(
                "dummy.mp4",
                data,
                self.graphic_paths,
                subtitle_config={"font": "MyFont", "fontsize": 50, "color": "red"}
            )

            self.mock_editor.TextClip.assert_called()
            args, kwargs = self.mock_editor.TextClip.call_args
            self.assertEqual(kwargs.get("font"), "MyFont")
            self.assertEqual(kwargs.get("fontsize"), 50)
            self.assertEqual(kwargs.get("color"), "red")

if __name__ == '__main__':
    unittest.main()
