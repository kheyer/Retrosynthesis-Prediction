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
