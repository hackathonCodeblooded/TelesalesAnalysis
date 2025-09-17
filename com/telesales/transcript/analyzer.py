from transformers import pipeline

# Sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

# Simple keyword-based action item extractor
def extract_action_items(text: str):
  action_keywords = ["follow up", "call back", "send", "schedule", "confirm", "share"]
  actions = [kw for kw in action_keywords if kw in text.lower()]
  return actions

def analyze_transcript(aligned_transcript):
  """
  Run NLP analysis (sentiment + action items) on aligned transcript.
  """
  analyzed = []
  for entry in aligned_transcript:
    sentiment = sentiment_model(entry["text"])[0]  # e.g. {'label': 'POSITIVE', 'score': 0.99}
    actions = extract_action_items(entry["text"])

    analyzed.append({
      "speaker": entry["speaker"],
      "text": entry["text"],
      "sentiment": sentiment["label"],
      "confidence": round(sentiment["score"], 2),
      "actions": actions
    })
  return analyzed
