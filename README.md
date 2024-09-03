# Active Inference Framework for Optimal Research Process Exploration

This repository implements an active inference framework designed to explore and exploit the optimal research process across different industries. The model demonstrates continuous learning and adaptation to changing environments, making it a powerful tool for understanding and optimizing research strategies in various industrial contexts.

## Table of Contents
1. [Introduction to Active Inference](#introduction-to-active-inference)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Generative Model](#generative-model)
4. [Implementation Details](#implementation-details)
5. [Results and Interpretation](#results-and-interpretation)

## Introduction to Active Inference

Active inference is a framework in computational neuroscience and artificial intelligence that unifies perception, learning, and decision-making under a single principle: the minimization of free energy. This principle is based on the idea that biological systems, including the brain, maintain a stable internal state by minimizing surprise (or uncertainty) about their environment.

In active inference, agents are not passive observers but active participants in their environment. They form beliefs about the world and then take actions to confirm or refute these beliefs. The process of belief updating and action selection is driven by the goal of minimizing expected free energy, which acts as a proxy for minimizing long-term surprise.

## Mathematical Foundations

### From Bayes' Rule to Free Energy Minimization

The free energy principle is derived from Bayes' rule, which describes how to update beliefs (posterior) based on new evidence (observations):

```
P(s|o) = P(o|s)P(s) / P(o)
```

Where:
- `s` represents hidden states of the world
- `o` represents observations

The concept of surprise is mathematically expressed as the negative log probability of an observation:

```
- log P(s|o) = - log P(o|s) - log P(s) + log P(o)
```

Minimizing surprise directly is intractable because it requires knowing all possible states. Instead, we approximate the true posterior `P(s|o)` with a variational distribution `Q(s)`:

```
- log P(o) = D_KL[Q(s) || P(s|o)] - ∫ Q(s) log [ P(o,s) / Q(s) ] ds
```

The second term on the right is defined as the negative free energy `F`. Since the KL divergence `D_KL` is non-negative, minimizing free energy indirectly minimizes surprise:

```
- log P(o) ≤ - F
```

### Free Energy as a Bound on Surprise

Free energy `F` can be expressed as:

```
F = E_Q[log Q(s) - log P(o,s)]
```

This can be further decomposed into:

```
F = D_KL[Q(s) || P(s)] - E_Q[log P(o|s)]
```

Where:
- The first term represents the divergence between the approximate posterior `Q(s)` and the prior `P(s)`.
- The second term represents the expected likelihood of the observations given the hidden states.

By minimizing free energy, the agent minimizes an upper bound on surprise, effectively learning about its environment and making decisions that reduce uncertainty.

## Generative Model

### Hierarchical Structure

The generative model in this framework is hierarchical, consisting of two levels:

1. **Top-Level (Industry Context)**: 
   - **Role**: Determines the current industry context in which the agent is operating.
   - **Function**: Provides a contextual prior that influences the lower-level processes, guiding the learning and decision-making towards industry-specific characteristics.

2. **Lower-Level (Research Process Optimization)**:
   - **Role**: Learns the optimal research processes within the context provided by the top level.
   - **Function**: Adapts to changes in the environment by updating its beliefs and refining its policies to optimize research processes in different industries.

This hierarchical structure allows the agent to contextualize its learning based on the industry it operates in, making it more efficient in exploring and exploiting the best research processes.

### Continuous Learning and Adaptation

The model demonstrates continuous learning by adapting to changes in what constitutes the best research process for different industries. As the environment evolves, the agent updates its beliefs and modifies its actions to maintain optimal performance.

## Implementation Details

The active inference agent utilizes the generative model to efficiently learn about its environment through action and policy selection. The implementation leverages key subroutines from the SPM12 package:

1. **`spm_MDP_VB`**: Performs variational Bayesian inference to update the agent's beliefs based on new observations.
2. **`spm_MDP_G`**: Generates outcomes (observations) based on the current state and the generative model.
3. **`spm_MDP_L`**: Computes the likelihood of observations given the current state.
4. **`spm_MDP_B`**: Updates the agent's beliefs about hidden states.
5. **`spm_MDP_P`**: Performs policy selection by evaluating the expected free energy of different action sequences.

### Agent Workflow

The agent follows these steps in each iteration:

1. **Observe the Environment**: The agent gathers new data from the environment.
2. **Update Beliefs**: The agent updates its internal beliefs using `spm_MDP_VB`.
3. **Select a Policy**: Based on its updated beliefs, the agent uses `spm_MDP_P` to select the policy that minimizes expected free energy.
4. **Execute an Action**: The agent carries out an action according to the selected policy.
5. **Update the Environment**: The environment state is updated in response to the agent's action.
6. **Repeat**: This cycle repeats, allowing the agent to continuously learn and adapt.

This process enables the agent to refine its understanding of the optimal research processes within each industry context while remaining flexible to environmental changes.

## Results and Interpretation

### Analysis of Figures in the `figure4` Folder

The figures in the `figure4` folder provide a visual representation of the agent's learning process, beliefs, actions, and policy selection across several trials:

1. **Belief Updates**: These figures show how the agent’s beliefs about the optimal research process evolve over time, represented by probability distributions. As the agent gathers more information, these distributions become more precise, reflecting increased confidence in its understanding.

2. **Action Selection**: The figures illustrate the agent's actions across trials, highlighting how it shifts from exploration (seeking new information) to exploitation (applying known optimal strategies) as its knowledge base grows.

3. **Policy Evaluation**: These plots depict the agent's policy evaluation process, where it selects actions based on expected free energy minimization. Policies that lead to more informative or beneficial outcomes are preferred.

4. **Adaptation to Changes**: Some figures may show how the agent adapts when the optimal research process for an industry changes. This is reflected in shifts in the agent's beliefs and corresponding changes in its actions and policy selections.

5. **Industry Context**: The figures also demonstrate the impact of the industry context on the agent’s behavior, showing how the top-level industry setting guides the lower-level learning processes.

By analyzing these figures, we can observe how the active inference framework enables the agent to efficiently explore and exploit optimal research processes across different industrial contexts, while also showcasing the agent's ability to adapt to dynamic environments.

## Conclusion

This project demonstrates the power of the active inference framework in optimizing research processes across various industries. By minimizing free energy and adapting to changing industry dynamics, the agent continuously learns and improves its strategies, offering a robust approach to exploration and exploitation in complex environments.
