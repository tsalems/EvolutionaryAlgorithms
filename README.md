
# Genetic and Evolutionary Algorithms
The Genetic and some Evolutionary Algorithms by Python with DEAP, Sklearn.

## Genetic Algorithm For Feature Selection
> Search the best feature subset for you classification model

## Description
Feature selection is the process of finding the most relevant variables for a predictive model. These techniques can be used to identify and remove unneeded, irrelevant and redundant features that do not contribute or decrease the accuracy of the predictive model.

In nature, the genes of organisms tend to evolve over successive generations to better adapt to the environment. The Genetic Algorithm is an heuristic optimization method inspired by that procedures of natural evolution.

In feature selection, the function to optimize is the generalization performance of a predictive model. More specifically, we want to minimize the error of the model on an independent data set not used to create the model.




## Dependencies
[Pandas](https://pandas.pydata.org/)

[Numpy](http://www.numpy.org/)

[scikit-learn](http://scikit-learn.org/stable/)

[Deap](https://deap.readthedocs.io/en/master/)


## Usage
1. Go to the repository folder
1. Run

Obs:
  - `n_population` and `n_generation` must be integers
  - You can go to the code and change the classifier so that the search is optimized for your classifier.

## Usage Example
Returns:
```
------------------------------1.dataset:landsatImg------------------------------
Accuracy with All features:
_Accuracy Model with KFold_
k=1 	 1.0 0.9445760303173851
k=2 	 1.0 0.946470866887731
k=3 	 1.0 0.9568924680246329
k=4 	 1.0 0.9469194312796209
k=5 	 1.0 0.9545023696682464
k=6 	 1.0 0.9469194312796209
k=7 	 1.0 0.9488151658767773
k=8 	 1.0 0.9535545023696682
k=9 	 1.0 0.9436018957345972
k=10 	 1.0 0.9497630331753555
		min		avg		max
TRAIN	1.0	1.0	1.0
TEST	0.9436018957345972	0.9492015194613634	0.9568924680246329

GA processing ...

gen	evals	std      	min     	avg     	max     
0  	5    	0.0457893	0.808555	0.894851	0.935932
1  	5    	0.0151121	0.899855	0.918961	0.935932
2  	5    	0.0138912	0.904215	0.919998	0.935932
3  	5    	0.0125713	0.904215	0.921552	0.935932
4  	5    	0.00900809	0.911354	0.926683	0.935932

Accuracy with Subset features:
_Accuracy Model with KFold_
k=1 	 1.0 0.9488394126006632
k=2 	 1.0 0.9403126480341071
k=3 	 1.0 0.9493131217432497
k=4 	 1.0 0.9511848341232227
k=5 	 1.0 0.9507109004739337
k=6 	 1.0 0.9502369668246445
k=7 	 1.0 0.942654028436019
k=8 	 1.0 0.9445497630331754
k=9 	 1.0 0.9545023696682464
k=10 	 1.0 0.9454976303317536
		min		avg		max
TRAIN	1.0	1.0	1.0
TEST	0.9403126480341071	0.9477801675269018	0.9545023696682464

Number of Features in Subset: 	9/14
Individual: [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1]

```
## Fonts
1. This repository was based on [GeneticAlgorithmForFeatureSelection](https://github.com/scoliann/GeneticAlgorithmFeatureSelection)
1. Examples for [DEAP](https://github.com/DEAP/deap).
1. Evolve a neural network with a genetic algorithm [neural-network-genetic-algorithm](https://github.com/harvitronix/neural-network-genetic-algorithm).
1. A number of re-sampling techniques commonly used in datasets showing strong between-class imbalance [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn).

#### Author: [Thai Nguyen](https://github.com/tsalems)