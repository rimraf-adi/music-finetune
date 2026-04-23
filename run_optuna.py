import optuna
import subprocess
import json
import os
import shutil
import sys

def objective(trial):
    # Suggest hyperparameters
    head_intermediate_size = trial.suggest_categorical("head_intermediate_size", [128, 256, 512])
    head_second_layer_size = trial.suggest_categorical("head_second_layer_size", [32, 64, 128])
    head_dropout_rate = trial.suggest_float("head_dropout_rate", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    lora_r = trial.suggest_categorical("lora_r", [4, 8, 16])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32])
    
    # We create a specific output dir for this trial so metrics don't overwrite
    output_dir = f"optuna_trial_{trial.number}"
    
    # Construct command
    # Use fewer epochs and smaller train pairs to speed up trials
    cmd = [
        sys.executable, "finetune_reward_model.py",
        "--head_intermediate_size", str(head_intermediate_size),
        "--head_second_layer_size", str(head_second_layer_size),
        "--head_dropout_rate", str(head_dropout_rate),
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--output_dir", output_dir,
        "--epochs", "2",
        "--max_train_pairs", "2000",
        "--batch_size", "16"
    ]
    
    print(f"\n--- Running trial {trial.number} ---")
    subprocess.run(cmd, check=False)
    
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    if not os.path.exists(metrics_path):
        # Return a terrible score if it failed
        return 0.0
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    best_val_acc = metrics.get("best_val_acc", 0.0)
    
    # Cleanup trial directory to save disk space
    try:
        shutil.rmtree(output_dir)
    except:
        pass
        
    return best_val_acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    best_params = study.best_params
    print("\nBest params found by Optuna:", best_params)
    
    with open("best_optuna_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
        
    print("Saved best_optuna_params.json")
    
    # Finally, train the main model using the best params
    print("\n--- Training final model with best params ---")
    final_output_dir = "final_reward_model_output"
    
    final_cmd = [
        sys.executable, "finetune_reward_model.py",
        "--config_json", "best_optuna_params.json",
        "--output_dir", final_output_dir,
        "--epochs", "10",  # Or could use default 30
        "--batch_size", "16"
    ]
    
    subprocess.run(final_cmd, check=True)
    print("\n--- Final training complete! ---")
    print(f"Final model saved in {final_output_dir}")
