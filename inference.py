import torch
import librosa
import argparse
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLING_RATE = 16000

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("model")
model = Wav2Vec2ForSequenceClassification.from_pretrained("model").to(device)

labels = sorted(["angry","happy","neutral","sad","surprise"])
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

def predict_one(file_path: str) -> str:
    # Single audio prediction, takes wav path and performs inference
    audio, sr = librosa.load(file_path, sr=SAMPLING_RATE)
    inputs = feature_extractor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()

    return id2label[pred_id]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file", help="audio file name", required=True)
    parser.add_argument("-s","--sample_rate",default=SAMPLING_RATE, help="Sampling rate", required=True)
    args = parser.parse_args()
    if not args.file:
        print("File name not provided")
    else:
        test_file = args.file
        prediction = predict_one(test_file)
        print(f" {test_file} -> Predicted emotion: {prediction}")
