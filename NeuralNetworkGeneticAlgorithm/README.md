# NeuralNetworkGeneticAlgorithm
## doc
### Brain

- build NN
- add gaussian on weights
- set, get weights
- @static cross brains (randomly get weights from parents) 

###  Creatures

- Brain <- Creatures
- record score
- @abstract get\_observations: get NN input, function give input
- @abstract think: get NN output as actions
- @abstract get\_reward, function give output as reward


### EvolveStrategy

- @static cross, use Brain.cross to cross creatures and return child
- @static mutate creatures(add gaussian to their Brain)
- @abstract evolve: implement how to select creatures to cross and mutate, according to their score, return total new cratures for next evolve, IO are CreaturesGroup

### CreaturesGroup
- List<Creatures>, EvolveStrategy <- CreaturesGroup
- sort according to score
- @final evolve: EvolveStrategy.evolve





