import unittest
from unittest.mock import MagicMock, patch
import sys
import importlib

class TestEditorOverlap(unittest.TestCase):
    def setUp(self):
        # Patch sys.modules to mock moviepy hierarchy
        # We need to set these before reloading src.editor
        self.patcher = patch.dict(sys.modules, {
            "moviepy": MagicMock(),
            "moviepy.editor": MagicMock(),
            "moviepy.video.fx.all": MagicMock(),
            "moviepy.audio.fx.all": MagicMock(),
            "moviepy.config": MagicMock(),
            "moviepy.video": MagicMock(),
            "moviepy.audio": MagicMock(),
        })
        self.patcher.start()

        # Reload src.editor so it imports the mocks we just set
        if 'src.editor' in sys.modules:
            import src.editor
            importlib.reload(src.editor)
        else:
            import src.editor

        self.Editor = src.editor.Editor

    def tearDown(self):
        self.patcher.stop()

    def test_graphic_overlap(self):
        editor = self.Editor()

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

        # We need to set the return value on the specific mock object that Editor imported
        # Since we reloaded Editor, it imported VideoFileClip from our mocked sys.modules["moviepy.editor"]
        mock_vfc_class = sys.modules["moviepy.editor"].VideoFileClip
        mock_vfc_class.return_value = mock_video

        # Mock ImageClip
        mock_img_class = sys.modules["moviepy.editor"].ImageClip
        mock_img_clip = MagicMock()
        mock_img_class.return_value = mock_img_clip
        mock_img_clip.set_start.return_value = mock_img_clip
        mock_img_clip.set_duration.return_value = mock_img_clip
        mock_img_clip.set_position.return_value = mock_img_clip
        mock_img_clip.resize.return_value = mock_img_clip

        # Mock CompositeVideoClip
        mock_cvc_class = sys.modules["moviepy.editor"].CompositeVideoClip
        mock_combined = MagicMock()
        mock_cvc_class.return_value = mock_combined
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
        self.assertEqual(mock_img_class.call_count, 2, "Graphic should be applied to both segments")

        # Verify first call (Seg 1)
        # Rel start = 4. Duration = 5-4 = 1.
        args1, _ = mock_img_clip.set_start.call_args_list[0]
        self.assertEqual(args1[0], 4.0)
        args1_d, _ = mock_img_clip.set_duration.call_args_list[0]
        self.assertEqual(args1_d[0], 1.0)

        # Verify second call (Seg 2)
        # Rel start = 0 (since it started before). Duration = 7-5 = 2.
        args2, _ = mock_img_clip.set_start.call_args_list[1]
        self.assertEqual(args2[0], 0.0)
        args2_d, _ = mock_img_clip.set_duration.call_args_list[1]
        self.assertEqual(args2_d[0], 2.0)

    def test_caption_overlap(self):
        editor = self.Editor()

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

        mock_vfc_class = sys.modules["moviepy.editor"].VideoFileClip
        mock_vfc_class.return_value = mock_video

        mock_txt_class = sys.modules["moviepy.editor"].TextClip
        mock_txt_clip = MagicMock()
        mock_txt_class.return_value = mock_txt_clip
        mock_txt_clip.set_start.return_value = mock_txt_clip
        mock_txt_clip.set_duration.return_value = mock_txt_clip
        mock_txt_clip.set_position.return_value = mock_txt_clip

        mock_cvc_class = sys.modules["moviepy.editor"].CompositeVideoClip
        mock_combined = MagicMock()
        mock_cvc_class.return_value = mock_combined
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

        self.assertEqual(mock_txt_class.call_count, 1, "Caption should be created for overlapping segment")

        # Rel start = 0 (started before). Duration = 8-5 = 3.
        args, _ = mock_txt_clip.set_start.call_args_list[0]
        self.assertEqual(args[0], 0.0)
        args_d, _ = mock_txt_clip.set_duration.call_args_list[0]
        self.assertEqual(args_d[0], 3.0)

if __name__ == '__main__':
    unittest.main()
