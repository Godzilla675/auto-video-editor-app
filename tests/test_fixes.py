import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock dependencies before imports
sys.modules["moviepy"] = MagicMock()
sys.modules["moviepy.editor"] = MagicMock()
sys.modules["moviepy.video.fx.all"] = MagicMock()
sys.modules["moviepy.audio.fx.all"] = MagicMock()
sys.modules["moviepy.config"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["numpy"] = MagicMock()

# Mock requests with a proper Exception class for RequestException
mock_requests = MagicMock()
class MockRequestException(Exception):
    pass
mock_requests.RequestException = MockRequestException
sys.modules["requests"] = mock_requests

import src.editor
import src.generator
from src.editor import Editor
from src.generator import Generator
import importlib

class TestFixes(unittest.TestCase):
    def setUp(self):
        # Ensure requests mock has the Exception class (in case overwritten)
        if not isinstance(sys.modules["requests"].RequestException, type):
             class MockRequestException(Exception): pass
             sys.modules["requests"].RequestException = MockRequestException

        importlib.reload(src.editor)
        importlib.reload(src.generator)

        self.mock_moviepy_editor = sys.modules["moviepy.editor"]
        self.mock_moviepy_editor.reset_mock()

        # Setup common mocks
        self.mock_video = MagicMock()
        self.mock_video.duration = 10.0
        self.mock_video.size = (1920, 1080)
        self.mock_video.w = 1920
        self.mock_video.h = 1080
        self.mock_video.subclip.return_value = self.mock_video
        self.mock_video.crossfadein.return_value = self.mock_video
        self.mock_video.audio = MagicMock() # Has audio by default

        self.mock_moviepy_editor.VideoFileClip.return_value = self.mock_video

        # Mock CompositeVideoClip to behave nicely
        def composite_side_effect(clips, size=None):
            m = MagicMock()
            m.duration = clips[0].duration if clips else 0
            m.crossfadein.return_value = m
            m.set_audio.return_value = m
            return m
        self.mock_moviepy_editor.CompositeVideoClip.side_effect = composite_side_effect

        self.mock_moviepy_editor.concatenate_videoclips.return_value = self.mock_video

        self.editor = Editor()

    def test_crossfade_capping(self):
        # Setup clips with short duration
        short_clip = MagicMock()
        short_clip.duration = 2.0
        short_clip.crossfadein.return_value = short_clip

        # Create a mock that returns list of clips
        # We need to simulate the 'clips' list being populated.
        # analysis_data with 2 segments
        analysis_data = {
            "segments": [
                {"start": 0, "end": 2.0},
                {"start": 2.0, "end": 4.0}
            ]
        }

        # Mock subclip to return short_clip
        self.mock_video.subclip.return_value = short_clip

        # Call edit with large crossfade
        self.editor.edit("dummy.mp4", analysis_data, {}, crossfade=5.0)

        # Verify crossfadein was called with capped value (2.0 - 0.1 = 1.9)
        # The first clip is not crossfaded, the second one is.
        # short_clip.crossfadein should be called with 1.9
        short_clip.crossfadein.assert_called_with(1.9)

    def test_intro_outro_logic(self):
        analysis_data = {"segments": [{"start": 0, "end": 5.0}]}

        # Mock _create_title_card
        with patch.object(self.editor, '_create_title_card') as mock_create_card:
            mock_card = MagicMock()
            mock_card.duration = 3.0
            mock_create_card.return_value = mock_card

            self.editor.edit("dummy.mp4", analysis_data, {}, intro_text="Intro", outro_text="Outro")

            # Check if title cards were created
            mock_create_card.assert_any_call("Intro", 3.0, (1920, 1080))
            mock_create_card.assert_any_call("Outro", 3.0, (1920, 1080))

            # concatenate_videoclips should receive 3 clips (intro, video, outro)
            args, _ = self.mock_moviepy_editor.concatenate_videoclips.call_args
            self.assertEqual(len(args[0]), 3)
            self.assertEqual(args[0][0], mock_card) # Intro
            self.assertEqual(args[0][-1], mock_card) # Outro

    def test_subtitle_background_box(self):
        analysis_data = {
            "segments": [{"start": 0, "end": 5.0}],
            "captions": [{"start": 1.0, "end": 4.0, "text": "Hello"}]
        }

        subtitle_config = {
            "box_color": "black",
            "box_opacity": 0.5
        }

        self.editor.edit("dummy.mp4", analysis_data, {}, subtitle_config=subtitle_config)

        # Check if ColorClip was used
        self.assertTrue(self.mock_moviepy_editor.ColorClip.called)
        call_args = self.mock_moviepy_editor.ColorClip.call_args
        self.assertEqual(call_args[1]['color'], "black")

        # Check if CompositeVideoClip was used for layers (Text + Box)
        # Note: Editor uses CompositeVideoClip for segments too, so we need to check calls carefully.
        # We expect a composite clip for the subtitle overlay
        # In edit(), if layers > 1 (which it will be: sub + composite_sub), it creates a composite.
        # But inside the caption loop, it creates a 'composite_sub' if box_color is set.
        # We can check if ColorClip was created.

    def test_audio_mix_check(self):
        analysis_data = {"segments": [{"start": 0, "end": 5.0}]}

        # Case 1: Video has no audio
        self.mock_video.audio = None
        self.mock_moviepy_editor.AudioFileClip.return_value.duration = 10.0

        with patch("os.path.exists", return_value=True):
            self.editor.edit("dummy.mp4", analysis_data, {}, music="music.mp3")

            # CompositeAudioClip should NOT be called because video has no audio
            # Instead set_audio should be called with music_clip directly
            self.mock_moviepy_editor.CompositeAudioClip.assert_not_called()

            # Verify set_audio was called
            self.assertTrue(self.mock_video.set_audio.called)

    def test_generator_retry_logic(self):
        # This test requires mocking requests, which we can do inside the test method
        with patch("src.generator.requests") as mock_requests:
            generator = Generator("token")

            # Simulate 503 then 200
            mock_response_503 = MagicMock()
            mock_response_503.status_code = 503
            mock_response_503.json.return_value = {"estimated_time": 0.1}

            mock_response_200 = MagicMock()
            mock_response_200.status_code = 200
            mock_response_200.content = b"fakeimage"

            mock_requests.post.side_effect = [mock_response_503, mock_response_200]

            with patch("src.generator.time.sleep") as mock_sleep:
                generator.generate("prompt")

                # Should have slept
                mock_sleep.assert_called()
                self.assertEqual(mock_requests.post.call_count, 2)

    def test_generator_retry_429(self):
        with patch("src.generator.requests") as mock_requests:
            generator = Generator("token")

            # Simulate 429 then 200
            mock_response_429 = MagicMock()
            mock_response_429.status_code = 429
            mock_response_429.headers = {"Retry-After": "1"}

            mock_response_200 = MagicMock()
            mock_response_200.status_code = 200
            mock_response_200.content = b"fakeimage"

            mock_requests.post.side_effect = [mock_response_429, mock_response_200]

            with patch("src.generator.time.sleep") as mock_sleep:
                generator.generate("prompt")

                mock_sleep.assert_called_with(1)
                self.assertEqual(mock_requests.post.call_count, 2)

    def test_generator_connection_error(self):
        generator = Generator("token")

        # Raise RequestException twice, then success
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake"

        # Use the exception class we defined
        exc = sys.modules["requests"].RequestException("Connection error")

        sys.modules["requests"].post.side_effect = [exc, exc, mock_response]

        with patch("src.generator.time.sleep") as mock_sleep:
            # We need to ensure we don't crash
            res = generator.generate("prompt")

            self.assertEqual(mock_sleep.call_count, 2)
            self.assertIsNotNone(res)

if __name__ == "__main__":
    unittest.main()
