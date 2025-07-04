# Audio-Classification-GTZAN

## Audio Genre Classification with CNN and GTZAN Dataset
This project implements model for music genres classification using spectograms and Convolutional neural network (CNN).

Used dataset [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), that contains 10 music genre.

## Goal
Train a model to classify audio files by genre, converting them into mel-spectrograms and using a deep convolutional neural network.

## Used technologies
- Python 3.10+
- PyTorch
- torchaudio
- matplotlib
- librosa

## Dataset
**GTZAN** — standart audio dataset, that contains:

- 10 genres (pop, rock, jazz, classical, blues, metal, hiphop, disco, reggae, country)
- 1000 audiofiles (30 sec each, 22050 Hz)
- Format: `.wav`

## Model architecture
- 5 `Conv2d` blocks with `BatchNorm`, `ReLU`, `MaxPool`
- `AdaptiveAvgPool2d` to reduce spatial dimension
- `Dropout` in classifier
- Activation: `ReLU`
- Classifier: `Linear` -> 10 classes

## Data transforms
- `MelSpectrogram`
- `AmplitudeToDB`
- `NormalizeDBRange(-60, 40)`
- `FrequencyMasking`, `TimeMasking`

## Results
- Best accuracy: **64.29% in test data**
- Optimizer: Adam (lr=0.001)
- Scheduler: ReduceLROnPlateau
- Loss: CrossEntropyLoss

## Possible improvments
- Fine-tuning models Wav2Vec2, PANNs, YAMNet
- Using more data (e.g. FMA)

## License
This project is licensed under the [MIT License](LICENSE).
