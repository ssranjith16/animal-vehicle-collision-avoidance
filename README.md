# Identify the Animal Challenge — Animal Detection & Classification

## Overview

This project is based on an animal image classification challenge focused on identifying animal species from wildlife images. The work explores deep learning for computer vision and also connects the problem to practical applications such as reducing animal-vehicle accidents.

## Challenge

The challenge provides 19,000 images covering 30 animal species. The task is to predict the probability for each animal class from a given image; the class with the highest probability is taken as the predicted animal.

The wildlife images present real-world difficulties including different poses, cluttered backgrounds, lighting and climate variation, viewpoints, occlusions, and visually similar animal classes.

## Dataset

- 13,000 training images
- 6,000 test images
- 30 animal species

The species include antelope, bat, beaver, bobcat, buffalo, chihuahua, chimpanzee, collie, dalmatian, german shepherd, grizzly bear, hippopotamus, horse, killer whale, mole, moose, mouse, otter, ox, persian cat, raccoon, rat, rhinoceros, seal, siamese cat, spider monkey, squirrel, walrus, weasel, and wolf.

## My Solution

I developed the solution using Google Colab and the fast.ai library, which is built on PyTorch.

### Approach

1. Used a pretrained ImageNet-based **ResNeXt-101 64x4d** model as the backbone.
2. Added two fully connected layers for the animal-classification task.
3. Selected a suitable learning rate for training.
4. Initially trained the added fully connected layers.
5. Fine-tuned the complete network and evaluated it on a validation set.
6. Applied data augmentation to improve model generalization.
7. Used **Test Time Augmentation (TTA)** during inference.
8. After achieving strong validation performance, trained the network again using all available training examples without a validation split.

## Workflow

```text
Animal Images
      ↓
Data Preparation & Augmentation
      ↓
Pretrained ResNeXt-101
      ↓
Fully Connected Classification Layers
      ↓
Fine-Tuning
      ↓
Validation / TTA
      ↓
Animal Class Probabilities
      ↓
Predicted Animal Class
```

## Technologies

- Python
- PyTorch
- fast.ai
- Deep Learning
- Computer Vision
- Image Classification
- Google Colab
- ResNeXt-101
- Data Augmentation
- Test Time Augmentation (TTA)

## Results

- **Validation Accuracy:** 97.5%
- **Leaderboard Position:** 4th place

## Files

```text
animal-vehicle-collision-avoidance/
├── README.md
├── notebooks/
│   ├── Identify_the_Animal_colab.ipynb
│   └── Identify_the_Animal_novalid_colab.ipynb
└── submission/
    └── subm01.csv
```

## Notebooks

- `notebooks/Identify_the_Animal_colab.ipynb` — main Colab implementation with training and validation workflow.
- `notebooks/Identify_the_Animal_novalid_colab.ipynb` — training workflow using all available training examples after model validation.

## Submission

The prediction output is stored in `submission/subm01.csv` using the class-probability submission format required by the challenge.

## Future Improvements

- Evaluate more recent object detection and image-classification architectures.
- Improve robustness for difficult wildlife conditions.
- Explore transfer learning with additional pretrained backbones.
- Add deployment for real-time animal detection from dashcam video.

## Disclaimer

This project is intended for educational and research purposes. The reported validation accuracy and leaderboard position are specific to the challenge setup and should not be interpreted as a production safety-system guarantee.
