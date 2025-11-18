# Sync Local OA_Evolution_Wildfires to Box
# Run this script whenever you want to backup your work to Box

$LOCAL_DIR = "C:\Users\smtku\OA_Evolution_Wildfires"
$BOX_DIR = "C:\Users\smtku\Box\OA_Evolution_Wildfires"

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Syncing to Box..." -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Git commit (optional - prompts user)
Write-Host "[Step 1] Git Commit" -ForegroundColor Yellow
$commitMessage = Read-Host "Enter commit message (or press Enter to skip Git commit)"

if ($commitMessage -ne "") {
    Set-Location $LOCAL_DIR
    git add .
    git commit -m "$commitMessage"
    Write-Host "Git commit created: $commitMessage" -ForegroundColor Green
} else {
    Write-Host "Skipped Git commit" -ForegroundColor Gray
}

Write-Host ""

# Step 2: Sync to Box
Write-Host "[Step 2] Syncing files to Box..." -ForegroundColor Yellow

# Use robocopy for efficient sync (only copies changed files)
robocopy $LOCAL_DIR $BOX_DIR /MIR /XD .git /XF .gitignore sync-to-box.ps1 /NP /NDL /NFL

if ($LASTEXITCODE -le 7) {
    Write-Host ""
    Write-Host "================================" -ForegroundColor Green
    Write-Host "SUCCESS! Synced to Box" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Warning: Sync completed with errors" -ForegroundColor Red
}

Write-Host ""
Write-Host "Local:  $LOCAL_DIR" -ForegroundColor Cyan
Write-Host "Box:    $BOX_DIR" -ForegroundColor Cyan
