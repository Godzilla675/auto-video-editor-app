import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import importlib

# Define mocks needed for src.editor if they don't exist
if "moviepy" not in sys.modules:
    sys.modules["moviepy"] = MagicMock()
if "moviepy.editor" not in sys.modules:
    sys.modules["moviepy.editor"] = MagicMock()
if "moviepy.video.fx.all" not in sys.modules:
    sys.modules["moviepy.video.fx.all"] = MagicMock()
if "moviepy.audio.fx.all" not in sys.modules:
    sys.modules["moviepy.audio.fx.all"] = MagicMock()
if "moviepy.config" not in sys.modules:
    sys.modules["moviepy.config"] = MagicMock()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.editor

class TestEditorConfig(unittest.TestCase):
    def setUp(self):
        # We must reload src.editor to ensure it captures the current mocks in sys.modules
        importlib.reload(src.editor)
        # We use the config object that src.editor actually imported
        self.mock_config = src.editor.mp_config

    def test_imagemagick_config(self):
        with patch.dict(os.environ, {"IMAGEMAGICK_BINARY": "/usr/bin/convert"}):
            self.mock_config.change_settings.reset_mock()
            src.editor.Editor()
            self.mock_config.change_settings.assert_called_with({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

    def test_imagemagick_no_config(self):
        with patch.dict(os.environ, {}, clear=True):
             if "IMAGEMAGICK_BINARY" in os.environ:
                 del os.environ["IMAGEMAGICK_BINARY"]

             self.mock_config.change_settings.reset_mock()
             src.editor.Editor()
             self.mock_config.change_settings.assert_not_called()

if __name__ == '__main__':
    unittest.main()
