# OA Evolution Wildfires Project

**GitHub:** https://github.com/Sumit79-coder/OA_Evolution_Wildfires

## Directory Structure

```
LOCAL (Work here - fastest):
C:\Users\smtku\OA_Evolution_Wildfires\
    ↓
GITHUB (Version control - accessible anywhere):
https://github.com/Sumit79-coder/OA_Evolution_Wildfires
    ↓
BOX (Additional backup):
C:\Users\smtku\Box\OA_Evolution_Wildfires\
```

## Workflow

### 1. **Work Locally**
- Open Jupyter notebooks from: `C:\Users\smtku\OA_Evolution_Wildfires\01_CMAQ_Analysis\notebooks\`
- All your changes are saved locally (fast, no sync delays)

### 2. **Commit to Git**
```bash
cd C:\Users\smtku\OA_Evolution_Wildfires
git add .
git commit -m "Description of changes"
```

### 3. **Push to GitHub**
```bash
git push
```

Your work is now backed up to the cloud and accessible from anywhere!

### 4. **Sync to Box (Optional)**
Double-click: `sync-to-box.bat`

Or run manually:
```bash
powershell -ExecutionPolicy Bypass -File sync-to-box.ps1
```

This will:
- Optionally create a Git commit
- Push to GitHub
- Copy all changes to Box
- Only sync files that changed (fast)

## Recovery

If you need to recover old versions:

### From Git (Local):
```bash
git log                    # See all commits
git show <commit-hash>     # View a specific commit
git checkout <commit-hash> <file>  # Restore a file
```

### From GitHub:
- Go to: https://github.com/Sumit79-coder/OA_Evolution_Wildfires
- Browse files and view commit history
- Access your code from any computer
- Clone to a new machine: `git clone https://github.com/Sumit79-coder/OA_Evolution_Wildfires.git`

### From Box:
- Go to https://app.box.com
- Navigate to the file
- Click "···" → "Version History"

## Tips

- **Commit often** - After each meaningful change (creates save points)
- **Push to GitHub** - Daily or when done working (cloud backup)
- **Sync to Box** - Optional, for additional backup layer
- **Don't work directly in Box folder** - Always use the local folder
- **Jupyter autosaves** - But still manually save important work
- **Access from anywhere** - Clone from GitHub on any computer

## Files

- `.gitignore` - Tells Git which files to ignore
- `sync-to-box.ps1` - PowerShell sync script
- `sync-to-box.bat` - Quick launcher for sync script
