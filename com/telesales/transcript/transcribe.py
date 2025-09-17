import whisper
import sys
import os

def transcribe_audio(audio_path, output_path="outputs/transcript.txt"):
  model = whisper.load_model("base")  # or "small", "medium", "large"
  result = model.transcribe(audio_path)

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  with open(output_path, "w") as f:
    f.write(result["text"])

  print(f"✅ Transcript saved to {output_path}")
  return result["text"]

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python transcriber.py <audiofile>")
    sys.exit(1)

  audio_file = sys.argv[1]
  transcribe_audio(audio_file)
