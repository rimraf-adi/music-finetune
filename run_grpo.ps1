$ErrorActionPreference = "Stop"
Write-Host "`n[3/3] Phase 2: GRPO Fine-Tuning..."
.venv\Scripts\python.exe -m training.grpo
if ($LASTEXITCODE -ne 0) {
    Write-Error "GRPO training failed!"
    exit $LASTEXITCODE
}
Write-Host "Pipeline execution completed successfully!"
