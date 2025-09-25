import os
import pandas as pd
from datasets import Dataset, Audio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer,DataCollatorWithPadding
import torch
print(torch.cuda.is_available())

# Parse RAVDESS and TESS datasets
def parse_ravdess(root):
    data = []
    emotion_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprise'
    }
    for root_dir, _, files in os.walk(root):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split('-')
                emotion = emotion_map.get(parts[2])
                if emotion:
                    data.append({'file_path': os.path.join(root_dir, file), 'label': emotion})
    return data

def parse_tess(root):
    data = []
    for root_dir, _, files in os.walk(root):
        for file in files:
            if file.endswith(".wav"):
                emotion = file.split('_')[-1].replace('.wav', '').lower()
                if emotion == "ps":
                    emotion = "surprise"
                data.append({'file_path': os.path.join(root_dir, file), 'label': emotion})
    return data

ravdess_data = parse_ravdess("FILL ME")
tess_data = parse_tess("FILL ME")
df = pd.DataFrame(ravdess_data + tess_data)
df = df[df['label'].isin(['happy', 'sad', 'angry', 'surprise', 'neutral'])]
df.to_csv("emotion_data.csv", index=False)

# Prepare Dataset
labels = sorted(df['label'].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

emotion_dataset = Dataset.from_pandas(df)
emotion_dataset = emotion_dataset.cast_column("file_path", Audio(sampling_rate=16000))
emotion_dataset = emotion_dataset.map(lambda x: {"label": label2id[x["label"]]})


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Preprocess with Feature Extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("r-f/jonatasgrosman/wav2vec2-large-xlsr-53-english")

def preprocess(batch):
    audio = batch["file_path"]["array"]
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    return {
        "input_values": inputs["input_values"][0],
        "attention_mask": inputs["attention_mask"][0] if "attention_mask" in inputs else None,
        "label": batch["label"]
    }

emotion_dataset = emotion_dataset.map(preprocess, remove_columns=["file_path", "label"],batched=False)

# Prepare Model
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "r-f/jonatasgrosman/wav2vec2-large-xlsr-53-english",
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
).to(device)

# Split dataset and fine-tune
emotion_dataset = emotion_dataset.train_test_split(test_size=0.2)
train_dataset = emotion_dataset["train"]
eval_dataset = emotion_dataset["test"]

def compute_metrics(pred):
    preds = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=-1)
    pred_ids = torch.argmax(preds, dim=-1)
    labels = torch.tensor(pred.label_ids)
    acc = (pred_ids == labels).float().mean()
    return {"accuracy": acc.item()}

training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    learning_rate=3e-5,
    num_train_epochs=2,
    logging_dir='./logs',
    save_total_limit=2,
    fp16=True,
)

print("Start training...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics
)

trainer.train()

# Save model
model.save_pretrained("model")
feature_extractor.save_pretrained("model")