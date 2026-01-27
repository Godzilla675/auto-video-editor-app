import unittest
from unittest.mock import MagicMock, patch
import sys

# Helper to mock a module structure
def mock_module(name):
    mod = MagicMock()
    sys.modules[name] = mod
    return mod

# Mock entire moviepy hierarchy
mock_moviepy = mock_module("moviepy")
mock_editor = mock_module("moviepy.editor")
mock_video = mock_module("moviepy.video")
mock_video_fx = mock_module("moviepy.video.fx")
mock_video_fx_all = mock_module("moviepy.video.fx.all")
mock_audio = mock_module("moviepy.audio")
mock_audio_fx = mock_module("moviepy.audio.fx")
mock_audio_fx_all = mock_module("moviepy.audio.fx.all")
mock_config = mock_module("moviepy.config")

# Now import Editor
from src.editor import Editor

class TestEditorOverlap(unittest.TestCase):
    def setUp(self):
        # Reset mocks between tests if needed, though we create fresh ones mostly
        pass

    def test_graphic_overlap(self):
        editor = Editor()

        # Mock video
        mock_video = MagicMock()
        mock_video.duration = 100.0
        mock_video.w = 100
        mock_video.h = 100

        def subclip_side_effect(start, end):
            m = MagicMock()
            m.duration = end - start
            m.w = 100
            m.h = 100
            return m
        mock_video.subclip.side_effect = subclip_side_effect

        with patch("src.editor.VideoFileClip", return_value=mock_video) as mock_vfc:
            with patch("src.editor.ImageClip") as mock_img_cls:
                mock_img_clip = MagicMock()
                mock_img_cls.return_value = mock_img_clip
                mock_img_clip.set_start.return_value = mock_img_clip
                mock_img_clip.set_duration.return_value = mock_img_clip
                mock_img_clip.set_position.return_value = mock_img_clip
                mock_img_clip.resize.return_value = mock_img_clip

                with patch("src.editor.CompositeVideoClip") as mock_cvc:
                    mock_combined = MagicMock()
                    mock_cvc.return_value = mock_combined
                    mock_combined.set_duration.return_value = mock_combined

                    # Segment 1: 0-5
                    # Segment 2: 5-10
                    # Graphic: start=4, duration=3 (4-7)
                    analysis_data = {
                        "segments": [
                            {"start": 0, "end": 5},
                            {"start": 5, "end": 10}
                        ],
                        "graphics": [
                            {"timestamp": 4, "duration": 3, "prompt": "test"}
                        ],
                        "captions": []
                    }
                    graphic_paths = {0: "dummy.png"}

                    with patch("os.path.exists", return_value=True):
                         editor.edit("dummy.mp4", analysis_data, graphic_paths)

                    # Check how many times ImageClip was created
                    # Should be twice: once for seg 1, once for seg 2
                    self.assertEqual(mock_img_cls.call_count, 2, "Graphic should be applied to both segments")

                    # Verify first call (Seg 1)
                    # Rel start = 4. Duration = 5-4 = 1.
                    args1, _ = mock_img_cls.return_value.set_start.call_args_list[0]
                    self.assertEqual(args1[0], 4.0)
                    args1_d, _ = mock_img_cls.return_value.set_duration.call_args_list[0]
                    self.assertEqual(args1_d[0], 1.0)

                    # Verify second call (Seg 2)
                    # Rel start = 0 (since it started before). Duration = 7-5 = 2.
                    args2, _ = mock_img_cls.return_value.set_start.call_args_list[1]
                    self.assertEqual(args2[0], 0.0)
                    args2_d, _ = mock_img_cls.return_value.set_duration.call_args_list[1]
                    self.assertEqual(args2_d[0], 2.0)

    def test_caption_overlap(self):
        editor = Editor()

        mock_video = MagicMock()
        mock_video.duration = 100.0
        mock_video.w = 100
        mock_video.h = 100

        def subclip_side_effect(start, end):
            m = MagicMock()
            m.duration = end - start
            m.w = 100
            m.h = 100
            return m
        mock_video.subclip.side_effect = subclip_side_effect

        with patch("src.editor.VideoFileClip", return_value=mock_video):
            with patch("src.editor.TextClip") as mock_txt_cls:
                mock_txt_clip = MagicMock()
                mock_txt_cls.return_value = mock_txt_clip
                mock_txt_clip.set_start.return_value = mock_txt_clip
                mock_txt_clip.set_duration.return_value = mock_txt_clip
                mock_txt_clip.set_position.return_value = mock_txt_clip

                with patch("src.editor.CompositeVideoClip") as mock_cvc:
                    mock_combined = MagicMock()
                    mock_cvc.return_value = mock_combined
                    mock_combined.set_duration.return_value = mock_combined

                    # Segment 1: 5-10
                    # Caption: start=4, end=8 (overlap 5-8)
                    analysis_data = {
                        "segments": [
                            {"start": 5, "end": 10}
                        ],
                        "graphics": [],
                        "captions": [
                            {"start": 4, "end": 8, "text": "hello"}
                        ]
                    }

                    with patch("os.path.exists", return_value=True):
                        editor.edit("dummy.mp4", analysis_data, {})

                    self.assertEqual(mock_txt_cls.call_count, 1, "Caption should be created for overlapping segment")

                    # Rel start = 0 (started before). Duration = 8-5 = 3.
                    args, _ = mock_txt_clip.set_start.call_args_list[0]
                    self.assertEqual(args[0], 0.0)
                    args_d, _ = mock_txt_clip.set_duration.call_args_list[0]
                    self.assertEqual(args_d[0], 3.0)

if __name__ == '__main__':
    unittest.main()
