# cars-neuralnetwork
A demonstration of neural network training using Python, Jupyter, Ketas. Included are two data sets with images from cars from various angles (front, left, right, frontleft, frontright, back). The images are taken from https://ai.stanford.edu/~jkrause/cars/car_dataset.html

Theses files have been used in a summer school webinar on machine learning by Sieuwert van Otterloo, Stefan Leijnen, Bert Wassink, Joost Krapels.

## Getting started


### Setup the data

Create the data and subdirectory

```bash
mkdir data
```

unzip the two data zip files into this directory

### (Optional) setup environment

If you have conda installed, you can create a seperate environment for the project using

Feel free to give the environment a different name than ml_webinar_hu if you like.

```bash
conda create -n ml_webinar_hu python=3.6
```

And activate the environment

```bash
conda activate ml_webinar_hu
```

### Install packages

Install the list of libraries into your environment

```bash
pip install -r requirements.txt
```

## Usage
Start jupyter and open to the desired notebook

```bash
jupyter notebook
```


