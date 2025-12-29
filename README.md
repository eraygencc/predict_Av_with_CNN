# predict_Av_with_CNN
Inferring dust extinction parameters from realistic galaxy images using CNNs

This project explores whether convolutional neural networks can learn physically meaningful dust extinction parameters from noisy simulated galaxy images.

The simulations are inspired by my PhD work on circumgalactic dust extinction and use GalSim to generate realistic imaging data including PSF convolution and noise.

The goal is to test if ML can be used to infer subtle dust extinction effects from galaxy image cutouts.

---

## Repository Structure

predict_Av_with_CNN/│
├── README.md
├── requirements.txt
│
├── notebooks/
│ └── predict_Av_with_CNN.ipynb # Experimentation and visualization
│
├── src/
│ ├── simulate_galaxy.py # galaxy image simulation with GalSim 
│ ├── cnn_model.py # CNN model definition
│ ├── dataset.py # Dataset generation and DataLoaders
│ └── train.py # Training and validation loop

Installing dependencies:
`pip install -r requirements.txt`

**Dependencies include:**
- `numpy`
- `torch`
- `torchvision`
- `galsim`
- `tqdm`

## CNN Model Overview

- **Input:** Single-channel grayscale galaxy image (e.g., 64×64 pixels)
- **Architecture:** 3 convolutional layers → ReLU → MaxPooling → AdaptiveAvgPool → Fully connected layer
- **Output:** Predicted dust extinction `tau` (scalar)
- **Loss:** Mean Squared Error (MSE) for regression

---

## Notes

- The galaxy simulation uses `GalSim` with these parameters:
  - Sersic profile 
  - Half-light radius 
  - Gaussian PSF 
  - Gaussian photometric noise
    
For simplicity, we show in the code only a set of parameters.

- The structure of the pipeline is:
  - `simulate_galaxy.py` → generates realistic galaxy images
  - `dataset.py` → wraps images in PyTorch `Dataset` and `DataLoader`
  - `cnn_model.py` → defines CNN
  - `train.py` → trains and evaluates the model

---

## License

MIT License

