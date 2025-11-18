# Sync Local OA_Evolution_Wildfires to GitHub and Box
# Run this script whenever you want to backup your work

$LOCAL_DIR = "C:\Users\smtku\OA_Evolution_Wildfires"
$BOX_DIR = "C:\Users\smtku\Box\OA_Evolution_Wildfires"

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Backup to GitHub & Box" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Git commit (optional - prompts user)
Write-Host "[Step 1] Git Commit" -ForegroundColor Yellow
$commitMessage = Read-Host "Enter commit message (or press Enter to skip)"

if ($commitMessage -ne "") {
    Set-Location $LOCAL_DIR
    git add .
    git commit -m "$commitMessage"
    Write-Host "Git commit created: $commitMessage" -ForegroundColor Green

    # Step 2: Push to GitHub
    Write-Host ""
    Write-Host "[Step 2] Pushing to GitHub..." -ForegroundColor Yellow
    git push
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Pushed to GitHub successfully" -ForegroundColor Green
    } else {
        Write-Host "Warning: GitHub push failed" -ForegroundColor Red
    }
} else {
    Write-Host "Skipped Git commit & GitHub push" -ForegroundColor Gray
}

Write-Host ""

# Step 3: Sync to Box
Write-Host "[Step 3] Syncing files to Box..." -ForegroundColor Yellow

# Use robocopy for efficient sync (only copies changed files)
robocopy $LOCAL_DIR $BOX_DIR /MIR /XD .git /XF .gitignore sync-to-box.ps1 /NP /NDL /NFL

if ($LASTEXITCODE -le 7) {
    Write-Host ""
    Write-Host "================================" -ForegroundColor Green
    Write-Host "SUCCESS! All backups complete" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Warning: Box sync completed with errors" -ForegroundColor Red
}

Write-Host ""
Write-Host "Local:   $LOCAL_DIR" -ForegroundColor Cyan
Write-Host "GitHub:  https://github.com/Sumit79-coder/OA_Evolution_Wildfires" -ForegroundColor Cyan
Write-Host "Box:     $BOX_DIR" -ForegroundColor Cyan
