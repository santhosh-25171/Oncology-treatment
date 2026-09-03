# Stage 02 — Deep Learning (Mini Demo)

**Mission:** Go from numbers to instincts — visually spot flooded streets from
camera images, and predict water level 3 hours ahead from a sequence of readings.

**Simplified for classroom pace:** Real CNNs/LSTMs need big libraries (TensorFlow/
PyTorch) and minutes of training. Here we use a small neural network (`MLPClassifier`
from scikit-learn) on tiny 8x8 synthetic "images" to teach the exact SAME workflow
(image in -> flood/clear out) in seconds. The real large-scale build (Day 3) will use
an actual CNN + LSTM/Transformer.

## Run order

| Script | Role | What it does |
|---|---|---|
| `01_data_engineer.py` | Data Engineer | Generates + labels mini synthetic street images, does simple augmentation |
| `02_eda_engineer.py` | EDA Engineer | Checks what pixel regions the brightness pattern comes from (saliency stand-in) |
| `03_dl_engineer.py` | DL Engineer | Trains a small neural net (CNN stand-in) + a trend predictor (LSTM stand-in) |
| `04_evaluation_engineer.py` | Evaluation Engineer | Confusion matrix — checks wet-road vs deep-flood confusion |
| `05_integration_engineer.py` | Integration Engineer | Combines vision + trend score into one alert API |
