from pyannote.audio import Pipeline
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, LocalEntryNotFoundError

def diarize_audio(audio_path: str):
  try:
    # Try loading the diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
  except GatedRepoError:
    print("\n❌ Access denied to 'pyannote/speaker-diarization'.")
    print("👉 Visit https://huggingface.co/pyannote/speaker-diarization and accept the user conditions.")
    return None
  except RepositoryNotFoundError:
    print("\n❌ Model repository not found on Hugging Face.")
    print("👉 Check the model name: pyannote/speaker-diarization")
    return None,
  except LocalEntryNotFoundError:
    print("\n❌ No valid Hugging Face token found.")
    print("👉 Run `huggingface-cli login` and make sure you have access to the model.")
    return None
  except Exception as e:
    print(f"\n❌ Unexpected error while loading diarization pipeline: {e}")
    return None

  try:
    # Run diarization
    diarization = pipeline(audio_path)
    return diarization
  except Exception as e:
    print(f"\n❌ Failed during diarization: {e}")
    return None
