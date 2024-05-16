from moviepy.editor import VideoFileClip
import io
from faster_whisper import WhisperModel

def extract_audio(video_path, audio_path):
    """Extract audio from a video file."""
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    video.close()



model_size = "small"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

def generate_subtitles(audio_path, subtitles_path):
    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    with open(subtitles_path, 'w') as f:
        counter = 1
        for segment in segments:
            start_time = segment.words[0].start
            end_time = segment.words[-1].end
            f.write(f"{counter}\n")
            f.write(f"{start_time} --> {end_time}\n")
            for word in segment.words:
                f.write(word.word)
            f.write("\n\n")
            print(segment.words)
            counter += 1
            
def main(video_path, audio_path, subtitles_path):
    """Main function to convert video to audio and generate subtitles."""
    #extract_audio(video_path, audio_path)
    generate_subtitles(audio_path, subtitles_path)

# Paths to the input video, output audio, and output subtitles
video_path = 'downloaded/The farming robots that will feed the world  Hard Reset.mp4'
audio_path = 'downloaded/output_audio.mp3'
subtitles_path = 'downloaded/output_subtitles.srt'

# Call the main function
main(video_path, audio_path, subtitles_path)
