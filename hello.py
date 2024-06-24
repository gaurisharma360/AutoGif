import whisper
import subprocess
import os
import cv2
import logging
from textblob import TextBlob

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2

# Initialize Whisper model
model = whisper.load_model("base")
logging.basicConfig(level=logging.DEBUG)

# Function to extract audio from video using FFmpeg
def extract_audio(video_path, audio_path):
    try:
        if os.path.exists(audio_path):
            logging.info(f"Audio file '{audio_path}' already exists. Skipping extraction.")
            return

        command = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting audio: {e}")

# Function to transcribe audio using Whisper
def transcribe_audio(audio_path):
    try:
        if model is None:
            logging.error("Whisper model not loaded correctly.")
            return None
        
        if not os.path.exists(audio_path):
            logging.error(f"Audio file '{audio_path}' not found.")
            return None

        result = model.transcribe(audio_path)
        return result["segments"]
    except Exception as e:
        logging.error(f"Error during transcription: {e}")

# Function to overlay text on video frames using OpenCV
def overlay_text_on_video(video_path, segments, output_video_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Adjust codec as needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)  # White

    segment_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        while segment_index < len(segments) and segments[segment_index]["end"] < current_time:
            segment_index += 1

        if segment_index < len(segments) and segments[segment_index]["start"] <= current_time < segments[segment_index]["end"]:
            text = segments[segment_index]["text"]
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 30  # Adjust this value to position text higher or lower

            # Draw text background
            margin = 10
            cv2.rectangle(frame, (text_x - margin, text_y - text_size[1] - margin), 
                          (text_x + text_size[0] + margin, text_y + margin), (0, 0, 0), cv2.FILLED)
            # Draw text
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Function to analyze sentiment using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to segment video based on sentiment changes
def segment_video(video_path, segments, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Error opening video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    os.makedirs(output_dir, exist_ok=True)

    segment_index = 0
    prev_sentiment = None
    out = None

    cap.set(cv2.CAP_PROP_POS_MSEC, segments[0]["start"] * 1000)

    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        sentiment = analyze_sentiment(text)

        logging.debug(f"Segment start: {start_time}, end: {end_time}, sentiment: {sentiment}")

        # Prepare VideoWriter for this segment
        output_file = os.path.join(output_dir, f'segment_{segment_index}.mp4')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        while cap.get(cv2.CAP_PROP_POS_MSEC) < end_time * 1000:
            ret, frame = cap.read()
            if not ret:
                break

            
            out.write(frame)

        out.release()
        segment_index += 1

    cap.release()

    logging.info(f"Segmented videos saved to {output_dir}")

# Function to combine video segments with original audio using FFmpeg
def combine_video_audio_segments(video_path, segments, output_dir):
    combined_files = []
    for i, segment in enumerate(segments):
        start_time = segment['start']
        end_time = segment['end']
        input_file = os.path.join(output_dir, f'segment_{i}.mp4')
        output_file = os.path.join(output_dir, f'segment_{i}_combined.mp4')

        try:
            command = [
                "ffmpeg",
                "-i", input_file,
                "-ss", str(start_time),
                "-to", str(end_time),
                "-i", video_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-strict", "experimental",
                "-map", "0:v:0",
                "-map", "1:a:0",
                output_file
            ]
            subprocess.run(command, check=True)
            combined_files.append(output_file)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error combining video and audio for segment {i}: {e}")

    logging.info(f"Combined video segments with audio saved to {output_dir}")

    # Convert combined video files to high-quality GIFs
    for combined_file in combined_files:
        gif_file = combined_file.replace(".mp4", ".gif")
        convert_video_to_high_quality_gif(combined_file, gif_file)

        
def convert_video_to_high_quality_gif(video_path, gif_path):
    try:
        # Create a temporary palette file
        palette_path = "palette.png"

        # First pass: generate the palette
        command_palette = [
            "ffmpeg",
            "-i", video_path,
            "-vf", "fps=10,scale=480:-1:flags=lanczos,palettegen",
            "-y", palette_path
        ]
        subprocess.run(command_palette, check=True)

        # Second pass: use the palette to create the GIF
        command_gif = [
            "ffmpeg",
            "-i", video_path,
            "-i", palette_path,
            "-filter_complex", "fps=10,scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse",
            "-y", gif_path
        ]
        subprocess.run(command_gif, check=True)

        # Remove the temporary palette file
        os.remove(palette_path)
        logging.info(f"Converted {video_path} to {gif_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting video to GIF: {e}")

def cleanup_files(output_dir):
    for file_name in os.listdir(output_dir):
        if not (file_name.endswith("_combined.mp4") or file_name.endswith(".gif")):
            os.remove(os.path.join(output_dir, file_name))
    logging.info(f"Cleanup completed. Only combined files are kept in {output_dir}")

# Function to process the video
def process_video(video_path):
    audio_path = "temp_audio.wav"  # Adjust this path as needed
    output_dir = "segmented_videos"
    output_video_path = "output_with_text.mp4"  # Adjust this path as needed

    # Extract audio from video
    extract_audio(video_path, audio_path)
    logging.info(f"Audio extracted to {audio_path}")

    # Transcribe the extracted audio
    segments = transcribe_audio(audio_path)
    if segments is not None:
        logging.info("Transcription completed.")
        
        # Overlay the transcript text on the video
        overlay_text_on_video(video_path, segments, output_video_path)
        logging.info(f"Output video saved to {output_video_path}")

        # Segment the video based on sentiment changes in the transcript
        segment_video(output_video_path, segments, output_dir)
        logging.info(f"Segmented videos saved to {output_dir}")

        # Combine video segments with original audio
        combine_video_audio_segments(video_path, segments, output_dir)
        logging.info("All segments combined with original audio.")

        # Clean up temporary files
        cleanup_files(output_dir)

   

    # Remove temporary audio file
    try:
        os.remove(audio_path)
    except FileNotFoundError:
        pass  # Ignore if file is already deleted

# Replace with your video path
video_path = "MadewithClipchamp.mp4"
process_video(video_path)
