from transcribe import transcribe_audio
from diarize import diarize_audio
from analyzer import analyze_transcript

def align_transcript(transcript, diarization):
  aligned = []
  for t in transcript:
    for d in diarization:
      if d["start"] <= t["start"] <= d["end"]:
        aligned.append({
          "speaker": d["speaker"],
          "start": t["start"],
          "end": t["end"],
          "text": t["text"]
        })
        break
  return aligned

if __name__ == "__main__":
  audio_path = "output.wav"

  # Step 1: Transcription
  transcript = transcribe_audio(audio_path)

  # Step 2: Diarization
  diarization = diarize_audio(audio_path)

  # Step 3: Alignment
  aligned = align_transcript(transcript, diarization)

  # Step 4: Analysis
  analyzed = analyze_transcript(aligned)

  for line in analyzed:
    print(f"[{line['speaker']}] {line['text']} "
          f"(Sentiment: {line['sentiment']} - {line['confidence']}, "
          f"Actions: {line['actions']})")
