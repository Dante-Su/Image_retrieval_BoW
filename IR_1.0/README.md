# Bag of words for image retrieval

*Author: DandingSu*

## Brief Intro

Python Implementation of Bag of Words for Image Retrieval using OpenCV and sklearn 

## Dependencies

**python=2.7**

```
pip install -r requirements.txt
```

## Code
- Training the codebook and quantization
```
python findFeatures.py -t dataset/training/
```

- Query a single image
```
python search.py -i dataset/train/ukbench00000.jpg
```

- Query all test image
```
python auto.py
```
