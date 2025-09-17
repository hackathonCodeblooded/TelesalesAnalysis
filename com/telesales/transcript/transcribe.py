import torch
import whisper


def transcribe_audio(audio_path: str, model_size="tiny"):
  """
  Transcribe audio using Whisper.
  Returns list of segments with text and timestamps.
  """
  device = "cpu"

  print(f"⚡ Using device: {device}")
  model = whisper.load_model(model_size, device=device)
  result = model.transcribe(audio_path)
  return result["segments"]
