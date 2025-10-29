# Enhancing Tropical Cyclone Intensity Estimation via Multi-Scale Spatial Feature Correlation

This project focuses on estimating the intensity of tropical cyclones using deep learning techniques. The model leverages multi-scale spatial feature correlation from satellite imagery to improve prediction accuracy.

## Dataset

This project uses the Tropical Cyclone for Image-to-intensity Regression (TCIR) dataset. It contains satellite images of tropical cyclones from four different channels (Infrared, Water Vapor, Visible, and Passive Microwave).

For more details and to download the dataset, please visit the official TCIR dataset page: [https://www.csie.ntu.edu.tw/~htlin/program/TCIR/](https://www.csie.ntu.edu.tw/~htlin/program/TCIR/)

## Code Structure

- `model.py`: Contains the implementation of the neural network model for cyclone intensity estimation.
- `main.py`: The main script to train and evaluate the regression model.
- `classify.py`: Contains code related to cyclone classification tasks.
- `main_class.py`: The main script for the classification model.

## Usage

To run the intensity estimation model, you can execute the `main.py` script. Make sure you have downloaded the TCIR dataset and placed it in the appropriate directory.

```bash
python main.py
```
