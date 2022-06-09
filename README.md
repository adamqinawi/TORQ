# Project TORQ
Designed and implemented by Adam Qinawi

# Operation
TORQ uses BOVW and KMeans clustering to train the model and KNN search to classify objects in the dataset.

# Installation
create Virtual Environment by using the following commands in windows command prompt:

```
	python -m venv .venv 
	.venv\Scripts\activate.bat  
	python -m pip install -r requirements.txt\
```
or the following commands in bash:

```
	python3 -m venv .venv
	source .venv/bin/activate
	python3 -m pip install -r requirements.txt
```
Then run main.py

# Requirements
Since this was originally for a school assignment, the following specifications were requested:
- Choose a suitable Feature Extraction algorithm, with some exclusions
- Implement classifier using Euclidean distance and KNN
- Print accuracy across K-values 1-9
- Dataset will be provided in the following structure:
	-{Type of set}/{Class}/{images}
	-Training set contains 787 images, validation set contains 105 images
