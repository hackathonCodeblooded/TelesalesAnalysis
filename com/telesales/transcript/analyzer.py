from transformers import pipeline
import spacy
import sys

# Load models
sentiment_pipeline = pipeline("sentiment-analysis")
nlp = spacy.load("en_core_web_sm")

def analyze_sentiment(text):
  return sentiment_pipeline(text)[0]

def extract_actions(text):
  doc = nlp(text)
  actions = []
  for sent in doc.sents:
    if any(token.lemma_ in ["schedule", "send", "call", "meet", "follow", "email"] for token in sent):
      actions.append(sent.text.strip())
  return actions

def analyze_transcript(file_path, output_path="outputs/analysis.txt"):
  with open(file_path, "r") as f:
    transcript = f.read()

  sentiment = analyze_sentiment(transcript)
  actions = extract_actions(transcript)

  with open(output_path, "w") as f:
    f.write("=== Sentiment ===\n")
    f.write(f"{sentiment}\n\n")
    f.write("=== Actions ===\n")
    for action in actions:
      f.write(f"- {action}\n")

  print(f"✅ Analysis saved to {output_path}")

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python analyzer.py <transcriptfile>")
    sys.exit(1)

  transcript_file = sys.argv[1]
  analyze_transcript(transcript_file)
