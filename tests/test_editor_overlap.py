import sys
from unittest.mock import MagicMock, Mock, patch
import unittest
import os
import bisect

# Mock heavy dependencies globally before importing src
mock_moviepy = MagicMock()
mock_moviepy_editor = MagicMock()
sys.modules["moviepy"] = mock_moviepy
sys.modules["moviepy.editor"] = mock_moviepy_editor
sys.modules["moviepy.config"] = MagicMock()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.editor
import importlib

from src.editor import Editor

class TestEditorOverlap(unittest.TestCase):

    def setUp(self):
        importlib.reload(src.editor)
        self.mock_moviepy_editor = sys.modules["moviepy.editor"]
        self.editor = Editor()

    def test_graphic_overlap(self):
        print("Testing Graphic Overlap...")

        # Mock VideoFileClip
        mock_clip = MagicMock()
        mock_clip.duration = 20.0
        mock_clip.w = 100
        mock_clip.h = 100
        mock_clip.subclip.return_value = mock_clip
        self.mock_moviepy_editor.VideoFileClip.return_value = mock_clip

        # Mock ImageClip
        mock_image_clip = MagicMock()
        mock_image_clip.set_start.return_value = mock_image_clip
        mock_image_clip.set_duration.return_value = mock_image_clip
        mock_image_clip.set_position.return_value = mock_image_clip
        mock_image_clip.resize.return_value = mock_image_clip
        self.mock_moviepy_editor.ImageClip.return_value = mock_image_clip

        # Mock CompositeVideoClip
        self.mock_moviepy_editor.CompositeVideoClip.return_value = MagicMock()

        # Mock concatenate_videoclips
        self.mock_moviepy_editor.concatenate_videoclips.return_value = MagicMock()

        # Define data
        # Segment: 5s - 10s
        # Graphic 1: Starts at 3s, duration 4s (Ends at 7s). Overlaps [5, 7]. Should be included.
        # Graphic 2: Starts at 6s, duration 2s (Ends at 8s). Overlaps [6, 8]. Should be included.
        # Graphic 3: Starts at 12s. Should NOT be included.
        analysis_data = {
            "segments": [{"start": 5, "end": 10}],
            "graphics": [
                {"timestamp": 3, "duration": 4, "prompt": "Graphic 1"},
                {"timestamp": 6, "duration": 2, "prompt": "Graphic 2"},
                {"timestamp": 12, "duration": 2, "prompt": "Graphic 3"},
            ]
        }
        graphic_paths = {0: "graphic1.png", 1: "graphic2.png", 2: "graphic3.png"}

        with patch('os.path.exists', return_value=True):
            self.editor.edit("dummy.mp4", analysis_data, graphic_paths)

        # Assertions
        # Check how many times ImageClip was instantiated or added to layers
        # In current logic, Graphic 1 (timestamp 3) is BEFORE start (5), so bisect_left(..., 5) skips it.
        # Graphic 2 (timestamp 6) is BETWEEN 5 and 10, so it is included.

        # We expect ImageClip to be called for Graphic 1 and Graphic 2.
        # But based on my analysis, Graphic 1 will be missed.

        # Let's count calls to ImageClip("graphic1.png") and ImageClip("graphic2.png")
        calls = self.mock_moviepy_editor.ImageClip.call_args_list
        print(f"ImageClip calls: {calls}")

        graphic1_called = any(call[0][0] == "graphic1.png" for call in calls)
        graphic2_called = any(call[0][0] == "graphic2.png" for call in calls)

        if not graphic1_called:
             print("FAILURE: Graphic 1 was NOT processed (overlap issue confirmed).")
        else:
             print("SUCCESS: Graphic 1 was processed.")

        if not graphic2_called:
             print("FAILURE: Graphic 2 was NOT processed.")

        self.assertTrue(graphic1_called, "Graphic 1 should be included because it overlaps the segment.")
        self.assertTrue(graphic2_called, "Graphic 2 should be included.")

if __name__ == '__main__':
    unittest.main()
