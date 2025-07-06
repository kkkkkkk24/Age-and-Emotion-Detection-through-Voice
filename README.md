# Age and Emotion Detection through Voice

## Description
Detects age from a male voice note, and if age > 60, also detects emotion. Rejects female voices.

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Place your trained models in the `model/` directory.
3. Run: `python main.py`

## Usage
- Click 'Upload Voice Note' to select an audio file.
- The app predicts age (and emotion if senior citizen) or rejects if not a male voice.