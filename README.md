# Speech Emotion Recognition

Part of my Master thesis, "Utilizing emotion recognition in audio DeepFake detection," was to create the model for emotion recognition. To do so, I have used the Wav2vec2 model on top of which I have added a classification layer. It is able to recognize 5 emotions: anger, happiness, neutral, sadness, and surprise. I have achieved 99.8\% accuracy while testing the model. My training script is adapted to use [RAVDESS Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) and [TESS Dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) datasets for training.


## Project Structure

.

├── train.py                 # Training script (RAVDESS + TESS)

├── inference.py             # Single file inference  

├── requirements.txt

└── README.md

---

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/ZofiaSzczepaniak/Speech-emotion-recognition.git
cd Speech_emotion_recognition
```

### (Optional) create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```
### Install dependencies
```bash
pip install -r requirements.txt
```

## Dataset Setup
Supported datasets:

    [RAVDESS Dataset]{https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio}

    [TESS Dataset]{https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess}

Download both datasets and update the dataset paths in train.py:

ravdess_data = parse_ravdess("/path/to/RAVDESS")
tess_data = parse_tess("/path/to/TESS")

## Training

Run training:

python train.py

This will:

    Parse RAVDESS + TESS datasets

    Filter for the following labels:

happy, sad, angry, surprise, neutral

Fine-tune wav2vec2-large-xlsr-53-english

    Save the model and feature extractor to fine-tuned-wav2vec2-emotion/

## Inference
Single audio file

python inference.py -f example.wav

Output:
example.wav -> Predicted emotion: happy

