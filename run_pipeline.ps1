$ErrorActionPreference = "Stop"

Write-Host "========================================="
Write-Host "Starting Complete RL Music Pipeline"
Write-Host "========================================="

Write-Host "`n[1/3] Phase 1: Pretraining CP Transformer..."
.venv\Scripts\python.exe -m training.pretrain
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pretraining failed!"
    exit $LASTEXITCODE
}

Write-Host "`n[2/3] Sensor Calibration: Training Reward Model..."
.venv\Scripts\python.exe -m training.train_reward
if ($LASTEXITCODE -ne 0) {
    Write-Error "Reward Model training failed!"
    exit $LASTEXITCODE
}

Write-Host "`n[3/3] Phase 2: GRPO Fine-Tuning..."
.venv\Scripts\python.exe -m training.grpo
if ($LASTEXITCODE -ne 0) {
    Write-Error "GRPO training failed!"
    exit $LASTEXITCODE
}

Write-Host "`n========================================="
Write-Host "Pipeline execution completed successfully!"
Write-Host "========================================="
