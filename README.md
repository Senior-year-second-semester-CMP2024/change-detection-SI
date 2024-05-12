# Change Detection

## Problem Statement

The objective is to detect changes between two images in a time series. Both input images share the same dimensions and represent the same location. The desired output is a binary image highlighting the areas of change, which could entail the introduction of new objects, the absence of previously present objects, or alterations in object positions.

## Prerequisites

- Python 3
- PyTorch
- tqdm
- OpenCV
- scikit-learn
- NumPy
- Matplotlib
- torchvision
- Jupyter Notebook

## Setup with Conda

```bash
conda create -n change-detection python=3.8
conda activate change-detection
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -y
conda install tqdm numpy matplotlib scikit-learn jupyter -y
pip install opencv-python
```

## Models

- Basic Unet (takes the difference between the two images -- before and after -- as input)
- Diff UNet (with Diffence Between input before and after)
- Siamese Nested UNet (UNet++)

## Dataset

Our dataset consists of 4868 samples captured from Egyptian lands before and after a certain period. We partitioned the dataset into 80% for training and 20% for validation purposes.

## Hyper Parameters

- Learning Rate: 0.001
- Learning Rate: StepLR with step size 10 and gamma 0.2
- Batch Size: 16
- Epochs: 50
- Optimizer: Adam
- Loss Function: BCEWithLogitsLoss

## Results Comparison

| Model                                         | Training Jaccard Score | Validation Jaccard Score |
| --------------------------------------------- | ---------------------- | ------------------------ |
| Basic UNet                                    | 0.85                   | 0.77                     |
| Diff UNet                                     | 0.86                   | 0.79                     |
| Siamese Nested UNet (UNet++)                  | 0.88                   | 0.81                     |
| Siamese Nested UNet (UNet++) with all dataset | 0.92                   | -                        |

## Testing

To test any model, use Predict.ipynb notebook and modify the following :

1. Change the model path
2. Import the model from `models` directory
3. Change the testset path (if needed)
4. Run the notebook

## Collaborators :handshake:

| [![bemoierian](https://avatars.githubusercontent.com/u/72103362?v=4)](https://github.com/bemoierian) | [![EngPeterAtef](https://avatars.githubusercontent.com/u/75852529?v=4)](https://github.com/EngPeterAtef) | [![markyasser](https://avatars.githubusercontent.com/u/82395903?v=4)](https://github.com/markyasser) | [![karimmahmoud22](https://avatars.githubusercontent.com/u/82693464?v=4)](https://github.com/karimmahmoud22) |
| ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| [Bemoi Erian](https://github.com/bemoierian)                                                         | [Peter Atef](https://github.com/EngPeterAtef)                                                            | [Mark Yasser](https://github.com/markyasser)                                                         | [Karim Mahmoud](https://github.com/karimmahmoud22)                                                           |
