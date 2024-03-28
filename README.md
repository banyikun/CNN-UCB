If you use the codes of this repository, please kindly cite following paper:

````
@article{ban2021convolutional,
  title={Convolutional Neural Bandit for Visual-aware Recommendation},
  author={Ban, Yikun and He, Jingrui},
  journal={arXiv preprint arXiv:2107.07438},
  year={2021}
}
````






# CNN-UCB

## Prerequisites: 

python 3.8.8

CUDA 11.2

torch 1.9.0

torchvision 0.10.0

sklearn 0.24.1

numpy 1.20.1

scipy 1.6.2

pandas 1.2.4

## Methods:

* cnn_ucb.py  -  Proposed algorithm CNN-UCB
* cnn-ts.py - Combine CNN with Thompson Sampling [Wang et al. 2021]
* cnn-epsilon.py - CNN with epsilon-greedy exploration strategy
* neural-epsilon.py - fully-connected neural network with epsilon-greedy exploration strategy
* neuralTS.py - Neural thompson sampling  [Zhang et al. 2020]
* neuralUCB.py - Neural UCB [Zhou et al. 2020]
* kernalUCB.py - Kernal UCB [Valko et al., 2013a]
* linUCB.py - LinUCB [Li et al., 2010]

* packages.py - all the needed packages
* load_data.py - load the datasets
* cnn_class.py - the CNN classes
* process_yelp.ipynb - extract features from Yelp dataset

## Datasets:
"mnist", "notmnist", "cifar10", "yelp"

## Run:
python "method" --dataset "dataset"

For example,   python cnn_ucb.py --dataset cifar10   ; python neuralUCB.py --dataset mnist


## Results and configurations are in results.pdf

## note:
To run dataset "notmnist", it is needed to download "imagedat.npy" and "labeldata.npy", and place them in the local folder.


To run dataset "yelp", it is needed to download " yelp_10000items_2000users_features.npy,  yelp_2000users_10000items.npy.zip, yelp_2000users_10000items_features.npy" and unzip "yelp_2000users_10000items.npy.zip" and place them in the local folder.

## Output: (number of round: regret, regret/number of round, loss)



