# Retrosynthesis Reaction Prediction with SMILES Augmentation

This repo details experiments on predicting retrosynthesis reactants from products using neural machine translation and SMILES based data augmentation.

## What is Retrosynthesis?

Organic synthesis is is the study of creating molecules through chemical reactions. Synthesis is often thought of as developing a pathway that will map some precursor compounds through a series of reactions to create a desired product. Good synthetic routes maximize speed, cost and yield while minimizing off target products. When an organic compound needs to be produced at scale, choosing the right synthetic route can have a major impact on the cost of production. Often, organic chemists find themselves in the position of planning reaction pathways out in reverse. Starting from the desired product molecule, how do we develop the most efficient synthesis pathway? This brings us to the problem of Retrosynthesis. Retrosynthesis is the process of taking some target molecule and working backwards iteratively to break the target molecule into simpler precursor molecules until all reactions start from a set of precursors that are cheap and widely available.

Retrosynthesis is hard. A major challenge is that the relationship between reactants and products has a many to one mapping. A specific target molecule can be made through multiple different synthetic routes. Consider the following product molecule:

![](media/product.png)

This molecule can be generated using either of the following reactant sets:

![](media/reactants.png)

In fact, every carbon to carbon bond in the product molecule is potentially a different synthetic route. For example:

![](media/retro_pathways.gif)
[source](https://pubs.acs.org/doi/10.1021/acscentsci.7b00303)

How do you choose which one is better? We can determine the criteria easily - we want the reaction that maximizes speed, yield, specificity (lack of off target products) and cost. Actually quantifying these metrics is another matter. If there is no literature on the specific reaction, it falls to an expert organic chemist. The chemist must use their knowledge of reaction mechanisms to estimate off target products. Yield must be determined experimentally, which takes time.

This process must be conducted iteratively at all levels of pathway planning. It's time consuming and relies on domain experts and their ability to keep thousands of reaction mechanisms in mind at all steps. Because of the challenge of this process, many computational tools have been developed to aid retrosynthesis planning.

## Computational Retrosynthesis

The difficulty of the Retrosynthesis problem has lead to a number of computational tools designed to assist chemists in planning synthetic routes. One way of developing a retrosynthesis algorithm is to use symbolic logic and expert designed rules. Rules based algorithms use specific reation rules defined by expert chemists in conjunction with a large database of known reactions to make retrosynthesis predictions. The problem with these systems is they generalize very poorly outside their specific rule set. This restricts the usefulness of rules based algorithms to simple targets, which are not particularly useful. 

Other algorithms use physical chemistry calculations to predict reaction energy barriers and likely products from first principals. This class of algorithms generalize much better to novel molecules, but are computationally prohibitive.

More recently, machine learning algorithms have been combined with rules based algorithms to improve performance, but have not escaped the generalization drawbacks of rules based systems. This has motivated fully machine learning based solutions to retrosyntehsis prediction.

## Deep Learning for Retrosynthesis

Deep learning was first applied to Retrosynthesis (to my knowledge) in 2017 by the [Liu et al](https://pubs.acs.org/doi/10.1021/acscentsci.7b00303). Liu et al framed the problem of Retrosynthesis prediction as a squence to sequence problem. The authors created a benchmark dataset of 50,000 reactant/product pairs with known reaction mechanisms. Liu et al used LSTM based sequence to sequence models and achieved 37.4% top-1 prediction accuracy compared to a rules based template matching model which achieved 35.4% top-1 accuracy.

Earlier this year (2019), [Lin et al](https://arxiv.org/abs/1906.02308) followed up on the results of Liu, using transformer based models for the sequence to sequence task. Lin et al achieved 54.6% top-1 accuracy on the benchmark dataset.

This repo shows how using specific data augmentation techniques for the sequence to sequence task can boost top-1 accuracy to 64%. But first, we need to discuss how exactly we can model reaction prediction as a sequence to sequence task.

## Text Representations of Chemical Reactions

In a sequence to sequence problem, our model takes as input a sequence of tokens and produces as output a new sequence of tokens. How do we represent chemical reactions as token sequence? We use SMILES strings.

[Simplified Molecular-Input Line-Entry System](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) strings or SMILES strings are a way of representing organic molecules as text strings in such a way that the structure of the molecule is captured in the SMILES string. For example:

![](media/smiles_example.png)

So we can represent the following reaction:

![](rxn1.png)

as `N#Cc1ccsc1N.O=[N+]([O-])c1cc(F)c(F)cc1F.C1CCOC1.[H-].[Na+] >> N#Cc1ccsc1Nc1cc(F)c(F)cc1[N+](=O)[O-]`

From a retrosynthesis standpoint, the product `N#Cc1ccsc1Nc1cc(F)c(F)cc1[N+](=O)[O-]` would be the input to the model, and the reactants `N#Cc1ccsc1N.O=[N+]([O-])c1cc(F)c(F)cc1F.C1CCOC1.[H-].[Na+]` would be the output.

## Developing a Dataset for Retrosynthesis

This repo shows results on the benchmark retrosynthesis dataset created by Liu et al. It's important to point out some of the concessions made in creating that dataset. There are many factors that go into *truly* predicting a synthetic pathway. A full synthesis prediction would include things like reagents, reaction conditions and yield. Liu et al simplified the Retrosynthesis problem by removing all reagents from the data, so the model is predicting only over main reactants.

![](media/reactants_reagents)

Reactions with multiple products are broken up so that each item in the dataset consists of one major product being formed by some reaction.

The dataset also classifies each reaction into ten different reaction categories.


| Reaction Class |              Reaction Name             |
|:--------------:|:--------------------------------------:|
|        1       |   heteroatom alkylation and arylation  |
|        2       |     acylation and related processes    |
|        3       |           C–C bond formation           |
|        4       |          heterocycle formation         |
|        5       |               protections              |
|        6       |              deprotections             |
|        7       |               reductions               |
|        8       |               oxidations               |
|        9       | functional group interconversion (FGI) |
|       10       |     functional group addition (FGA)    |

A token representing the reaction type is prepended to the product SMILES as part of the input to the model. This greatly constrains the set of possible reactants and makes the prediction problem easier.

After all these stages of filtering, a final datapoint might look like this:

Input: `<RX_6> C/C=C/c1cc(C(=O)O)c(F)cc1OCC12CC3CC(CC(C3)C1)C2`

Output: `C/C=C/c1cc(C(=O)OC(C)(C)C)c(F)cc1OCC12CC3CC(CC(C3)C1)C2`

## SMILES-Based Data Augmentation

Data augmentation is a collection of methods that allow us to modify our dataset to appear to have more unique data points than it really does. Data augmentation techniquesfor images such as flipping, rotating or cropping images, are widely used in computer vision. Representing molecules as SMILES strings allows us to use specific data augmentation strategies.

Organic molecules are commonly represented as graphs. SMILES strings contain all the same information as a molecular graph, and one can easily convert between representations. Similar to how a specific graph can have multiple valid representations, a molecule can have multiple valid SMILES strings. For example, the following SMILES all represent the same molecule:

    c1([C@@]2(OC(=O)CC)CC[NH+](C)C[C@H]2CC=C)ccccc1
    c1ccc([C@]2(OC(=O)CC)[C@H](CC=C)C[NH+](C)CC2)cc1
    c1ccccc1[C@@]1(OC(CC)=O)CC[NH+](C)C[C@H]1CC=C
    O=C(CC)O[C@]1(c2ccccc2)CC[NH+](C)C[C@H]1CC=C
    C[NH+]1CC[C@](OC(=O)CC)(c2ccccc2)[C@H](CC=C)C1
    C1[C@@](c2ccccc2)(OC(CC)=O)[C@H](CC=C)C[NH+](C)C1
    [C@]1(c2ccccc2)(OC(=O)CC)CC[NH+](C)C[C@H]1CC=C
    c1([C@@]2(OC(CC)=O)CC[NH+](C)C[C@H]2CC=C)ccccc1
    [C@@H]1(CC=C)C[NH+](C)CC[C@]1(OC(=O)CC)c1ccccc1
    c1cccc([C@@]2(OC(=O)CC)CC[NH+](C)C[C@H]2CC=C)c1



Typically a SMILES representation of a molecule is put through a canonicalization algorithm to set a specific standard representation for each molecule.

We can use different valid SMILES representations to augment our dataset. This technique was proposed by [Bjerrum](https://arxiv.org/abs/1703.07076) and implemented in this [repo](https://github.com/EBjerrum/SMILES-enumeration).

This technique was used by [Schwaller et al](https://arxiv.org/abs/1811.02633) in the task of synthesis prediction and showed an overall accuracy improvement.
