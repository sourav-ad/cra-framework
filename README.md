# Causal Off-Policy Evaluation for Sequential Decisions: A Doubly Robust Framework

We aim to introduce the Causal Resource Added (CRA) metric for policy intervention strategies within sequential decision making environments that tell us how much improvement (or deterioration) does an intended intervention offer over a hypothetical baseline intervention, while taking causality into account. 

To estimate CRA, we present a *doubly robust*  augmented inverse probability weighted (AIPW) estimator. Theoretically, for our estimator, we prove sequential double robustness and invariance to additive time-dependent shifts.

We present a simulation study to demonstrate the same. The folder `src` contains the source codes and `notebooks` contains the implementation and simulation. 