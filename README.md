# NTU-CV2024 FINAL
![](https://github.com/leolinpotato/door-status-tracker/blob/main/demo.gif)

## Table of Contents
- [Introduction](#introduction)
- [Proposed Method](#proposed-method)
- [Techniques](#techniques)
- [Installation](#installation)
- [Usage](#usage)
- [Member](#member)

## Introduction
This is the final project of the Computer Vision Course (Spring 2024, National Taiwan University) sponsored by Vivotek.

## Proposed Method
Since the relative position of camera and the bus is fixed, we adapt a background subtraction related method. We first apply frame by frame absolute difference to find out moving objects, then try to focus on door movements to identify the state of the door.

The moving objects can be classified into the following categories: people, bags, windows, and door. For people and bags, we import the YOLOv8 model to detect them and filter out their movements. For windows, we design a two pass method to tackle them. Since the lighting condition varies significantly on the windows, their average value after applying absolute difference tend to be high. Therefore, in the first pass, we will first calculate the average value of absolute difference in each pixel. Then the average value will later be used as mask to mitigate the noise of flickering windows.

After those processing, we can obtain a clean frame which only reflects the door movements. Since the door only moves during opening and closing, we can identify door state by simply summing up the pixel value of the processed frame. With value higher than a certain threshold we'll define them as opening and closing.

## Techniques
### Preprocess
- Two pass method to obtain average pixel map

### Detection
- YOLOv8
- Frame by frame absolute difference
- Mask out middle area for noise reduction

### Postprocess
- Clipping value of outliers by upper bound and lower bound
- Gaussian filter to smooth the curve
- Clustering on candidates in case making duplicate prediction on single opening/closing

## Installation
- python==3.12.3
### TAs
```sh
# Create virtual environment
conda create -n cv_final python=3.12.3

# Activate the virtual environment
conda activate cv_final

# Install dependencies
pip install -r requirements.txt
```

### Others
```shell
# Clone the repository
git clone https://github.com/leolinpotato/door-status-monitoring

# Navigate to the project directory
cd door-status-monitoring

# Create virtual environment
conda create -n cv_final python=3.12.3

# Activate the virtual environment
conda activate cv_final

# Install dependencies
pip install -r requirements.txt
```

## Usage
```shell
# Run
python guess.py --dir "directory of your test videos" --output "name of output json file"
```

## Result
| Public Recall | Public Precision | Total Recall | Total Precision |
|----------|----------|----------|----------|
| 100% | 100% | 90% | 85.7% |

## Member
B10902024 林宸宇  
B10902060 翁菀羚  
B10902126 陳致翰  
B10902138 陳德維
