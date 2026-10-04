# End-to-End Symbolic Music Generation Pipeline (ATEPP Dataset)

## Overview
We are implementing a closed-loop control framework for symbolic music generation, as formulated in the associated paper. This framework uses a causal Transformer (the Plant), a pretrained MidiBERT reward model (the Sensor), and Group Relative Policy Optimization (the Controller). 

## 1. Data Preparation
- **Objective**: Tokenize the 12,142 MIDI files from the ATEPP dataset into Compound Word (CP) tokens.
- **Process**: Parse each MIDI file into note objects, quantize to 16th-note resolution, and encode them into a 4-dimensional tuple (Bar, Position, Pitch, Duration).
- **Output**: The tokenized sequences are randomly shuffled and split into a 75% pretraining set and a 25% GRPO fine-tuning set.

## 2. Phase 1: Pretraining the CP Transformer (Plant)
- **Objective**: Learn the open-loop dynamics of musical sequences using Maximum Likelihood Estimation (MLE).
- **Model**: A causal Transformer with 8 layers, 8 heads, and ~26M parameters.
- **Process**: 
  - Train on the 75% pretraining split using next-token prediction across all four CP attributes.
  - Optimize using AdamW with cosine annealing.
- **Output**: The model reaches a steady-state equilibrium and weights are saved as the foundation for the active policy and frozen reference policy.

## 3. Sensor Calibration: MidiBERT Reward Model
- **Objective**: Train a scalar reward model that acts as the "Sensor" to measure the quality of generated sequences.
- **Model**: A 12-layer bidirectional MidiBERT-Piano model with an appended MLP head, using LoRA for parameter efficiency.
- **Process**: 
  - Evaluate synthetic preference pairs using a Bradley-Terry ranking loss.
- **Output**: A calibrated scalar quality metric that acts as our objective function for GRPO.

## 4. Phase 2: GRPO Fine-Tuning (Controller)
- **Objective**: Fine-tune the CP Transformer using reinforcement learning to maximize the MidiBERT reward while staying anchored to the pretraining distribution.
- **Process**:
  - Load the pretraining weights into the `active_policy` and the frozen `ref_policy`.
  - Load the trained `RewardModel` to evaluate generation rollouts.
  - Generate multiple completions per prompt.
  - Compute group-relative advantages.
  - Optimize the active policy using the GRPO clipped surrogate objective.
  - Regularize using a Kullback-Leibler (KL) divergence penalty against the `ref_policy` and an entropy floor to prevent mode collapse.
- **Metrics to Extract**:
  - `loss_total`, `loss_bar`, `loss_position`, `loss_pitch`, `loss_duration` (Pretraining)
  - `eval_reward` (Mean, Std, Max)
  - `kl_divergence`
  - `entropy`
  - `advantage` statistics

## 5. Automation Strategy
Since the tokenization step is running sequentially on 12K files, we will orchestrate the pipeline via a robust master shell script. The script will automatically trigger Pretraining -> Reward Model -> GRPO in sequence. Checkpoints will be aggressively logged, and tensorboard/json logging will capture the detailed metrics for evaluation.
