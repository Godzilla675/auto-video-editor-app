# Automated Video Editor

This application automates the video editing process using advanced AI models. It runs entirely on GitHub Actions, taking a raw video input and producing a polished, edited video with cuts, captions, and generated graphics.

## Features

- **Intelligent Editing**: Uses **Gemini 3 Pro** (via proxy) to analyze video content and transcription to determine the best cuts and transitions.
- **Transcription**: Utilizes **OpenAI Whisper** for accurate audio transcription.
- **Graphics Generation**: automatically generates illustrative graphics using **FLUX.1-dev** (Hugging Face) when the speaker explains concepts.
- **Automated Workflow**: Triggers automatically on video upload or via a manual dispatch with a URL.

## Setup Instructions

### 1. GitHub Secrets configuration

To enable the AI capabilities, you must configure the following secrets in your GitHub repository:

1. Go to your repository on GitHub.
2. Navigate to **Settings** > **Secrets and variables** > **Actions**.
3. Click **New repository secret**.
4. Add the following secrets:

| Name | Value |
|------|-------|
| `GEMINI_API_KEY` | `<YOUR_GEMINI_API_KEY>` |
| `HF_TOKEN` | `<YOUR_HUGGING_FACE_TOKEN>` |

*Note: These keys are provided for this specific project configuration.*

### 2. Permissions

Ensure your GitHub Actions have read/write permissions if you plan to push changes, although this workflow primarily uses Artifacts for output.

## How to Use

### Method 1: URL Trigger (Recommended)

1. Go to the **Actions** tab in your repository.
2. Select the **Automated Video Editor** workflow on the left.
3. Click the **Run workflow** button.
4. Paste a direct URL to a video file (e.g., `https://example.com/my_video.mp4`).
5. Click **Run workflow**.

### Method 2: File Upload

1. Push an `.mp4` file to the repository.
2. The workflow will automatically trigger, process the video, and generate the result.

## Retrieving the Result

1. When the workflow run finishes, click on the run to view details.
2. Scroll down to the **Artifacts** section.
3. Download the `final-video` artifact, which contains the edited `final_video.mp4`.

## Local Development

If you wish to run the application locally:

### Prerequisites
- Python 3.10 or higher.
- **FFmpeg**: Required for video processing.
  - Ubuntu: `sudo apt-get install ffmpeg`
  - MacOS: `brew install ffmpeg`
- **ImageMagick**: Required for caption generation.
  - Ubuntu: `sudo apt-get install imagemagick`
  - **Important**: You may need to edit `/etc/ImageMagick-6/policy.xml` to allow read/write permissions for PDF/Text operations.

### Installation

```bash
# Clone the repository
git clone <repo_url>
cd <repo_name>

# Install dependencies
pip install -r requirements.txt
```

### Running the App

Set the environment variables and run the main script:

```bash
export GEMINI_API_KEY="<YOUR_API_KEY>"
export HF_TOKEN="<YOUR_HF_TOKEN>"

# Run with a video URL
python -m src.main --url "https://example.com/video.mp4" --output my_edit.mp4

# OR run with a local file
python -m src.main --video "input.mp4" --output my_edit.mp4
```

## Project Structure

- `src/`: Source code modules.
  - `transcriber.py`: Whisper integration.
  - `analyzer.py`: Gemini AI interface.
  - `generator.py`: Hugging Face graphics generation.
  - `editor.py`: MoviePy editing logic.
  - `main.py`: CLI entry point.
- `.github/workflows/`: CI/CD configuration.
- `tests/`: Unit tests.
