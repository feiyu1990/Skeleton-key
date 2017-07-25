# Skeleton-key
The implementation of the model in paper "[Skeleton Key: Image Captioning by Skeleton-Attribute Decomposition](http://acsweb.ucsd.edu/~yuw176/skeleton-key.html)"

# # # Prerequisite
The model uses tensorflow, and the preprocessing of the captions requires [Stanford NLP Core](https://stanfordnlp.github.io/CoreNLP/) and you need to download COCO dataset first.

# # # Dataset preprocessing

Use create_data.py to create the skeleton-attribute dataset from COCO.


# # # Test
Download the pre-trained model at [Drive](https://drive.google.com/open?id=0BxguZu5SanNxNUdnYkV6Z3h5aUE.), and put the model under ./model
Use run_inference.py to test the model on the 5000-split test set.

