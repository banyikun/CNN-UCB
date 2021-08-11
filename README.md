# CNN-UCB

## Prerequisites: latest pytorch and cuda

## Methods:

* cnn_ucb.py  -  Proposed algorithm CNN-UCB
* cnn-epsilon.py - CNN with epsilon-greedy exploration strategy
* neural-epsilon.py - fully-connected neural network with epsilon-greedy exploration strategy
* neuralTS.py - Neural thompson sampling  [Zhang et al. 2020]
* neuralUCB.py - Neural UCB [Zhou et al. 2020]
* kernalUCB.py - Kernal UCB [Valko et al., 2013a]
* linUCB.py - LinUCB [Li et al., 2010]

* packages.py - all the needed packages
* load_data.py - load the datasets
* cnn_class.py - the CNN classes

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



