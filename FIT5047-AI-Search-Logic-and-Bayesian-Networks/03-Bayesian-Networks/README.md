# Bayesian Network: Fraud Detection and Decision Analysis

The assignment uses **Netica 7.01** to model card transaction fraud with prior probabilities, conditional probability tables (CPTs), evidence updates and decision utilities.

## Files

- `Q1.neta` — original Netica binary network; **requires Netica** to inspect/edit and is not previewable on GitHub.
- `Q2.dne` — text-based Netica network representation, viewable directly on GitHub.

## Q2 network (from the supplied DNE)

- Chance nodes include `Trav` (travel), `Fraud`, `OC`, `FP`, `IP` and `CRP`.
- Decision node `B` depends on `FP`, `CRP` and `IP`.
- Utility node `U` depends on `Fraud` and `B`.
- The model includes a prior fraud probability of **0.43%**, calculated from the supplied travel-conditioned probabilities.

The supplied PDF discusses evidence updates, including foreign purchases and online purchases. No independent re-run or Netica validation has been performed.
