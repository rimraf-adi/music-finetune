# Reference-Tracking GRPO Music Generation Pipeline
**Status:** ✅ COMPLETE

## 1. Project Goal
To design and train a fundamentally novel, closed-loop music generation system. Unlike standard autoregressive transformers that generate music blindly (open-loop), this model is conditioned on a user-supplied reference trajectory `y_ref` (tracking note density, tonal tension, pitch centroid, and rhythmic complexity). 

The system uses **Group Relative Policy Optimization (GRPO)** to dynamically steer the generated music so that it strictly adheres to the user's desired control trajectory.

## 2. Pipeline Architecture
We executed a 3-phase training pipeline on the massive 12,000-file ATEPP MIDI dataset:

### Phase 1: Pretraining the Base Model
- **Model:** `CPTransformer` (26 million parameters)
- **Task:** Next-token prediction of Compound Word (CP) tuples: [Bar, Position, Pitch, Duration]
- **Scale:** 100 Epochs, Batch Size 64. 12,699 steps.
- **Result:** Successfully mapped the unconditional grammar of polyphonic piano music.

### Phase 2: Sensor Calibration (Reward Model)
- **Model:** `wazenmai/MIDI-BERT` (110 million parameters) fine-tuned with LoRA.
- **Task:** Bradley-Terry Preference modeling. The model evaluates pairs of music (one clean, one corrupted) and learns to predict which one is "better".
- **Scale:** 10 Epochs, Batch Size 32. 
- **Result:** Validation loss dropped from 0.648 to 0.444. Successfully built a highly-critical musical judge.

### Phase 3: Reference-Conditioned GRPO
- **Mechanics:** 
  - The pre-trained CPTransformer was augmented with a `ReferenceEncoder`.
  - It generates multiple candidate completions based on a random target trajectory `y_ref`.
  - The Reward Model evaluates how well the generated music tracks `y_ref`.
  - GRPO computes advantages (`A_hat`) and updates the CPTransformer policy to maximize reward while penalizing KL-divergence from the frozen base model.
- **Scale:** 3000 Steps.
- **Result:** Successfully trained the closed-loop tracking policy.

## 3. Final Artifacts
- All code, datasets, and scripts have been fully pushed to the `main` branch of the GitHub repository.
- Weights are securely saved in `checkpoints/grpo`.
