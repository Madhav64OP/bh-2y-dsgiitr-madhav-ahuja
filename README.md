# Real-Time Emotion Detection from Speech

This project uses machine learning to detect emotions in speech. The pipeline includes:

1. **Preprocessing**: Extracts Mel-frequency cepstral coefficients (MFCC) features.
2. **Model Training**: A trained model classifies audio data into predefined emotions.
3. **Real-Time Detection**: Captures live audio and predicts the emotion.

## Features
- Real-time audio recording and processing.
- Emotion detection using a pre-trained model.

## How to Use
1. Install dependencies using `pip install -r requirements.txt`.
2. Run the script with `python emotion_detection.py`.
3. Provide live audio input when prompted.

## Files
- `emotion_detection.py`: Main Python script.
- `MODEL.json`: Contains model architecture and parameters.
