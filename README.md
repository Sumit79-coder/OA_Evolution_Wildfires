# OA Evolution Wildfires Project

## Directory Structure

```
LOCAL (Work here):
C:\Users\smtku\OA_Evolution_Wildfires\

BACKUP (Auto-synced):
C:\Users\smtku\Box\OA_Evolution_Wildfires\
```

## Workflow

### 1. **Work Locally**
- Open Jupyter notebooks from: `C:\Users\smtku\OA_Evolution_Wildfires\01_CMAQ_Analysis\notebooks\`
- All your changes are saved locally (fast, no sync delays)

### 2. **Commit to Git (Recommended)**
```bash
cd C:\Users\smtku\OA_Evolution_Wildfires
git add .
git commit -m "Description of changes"
```

### 3. **Sync to Box**
Double-click: `sync-to-box.bat`

Or run manually:
```bash
powershell -ExecutionPolicy Bypass -File sync-to-box.ps1
```

This will:
- Optionally create a Git commit
- Copy all changes to Box
- Only sync files that changed (fast)

## Recovery

If you need to recover old versions:

### From Git:
```bash
git log                    # See all commits
git show <commit-hash>     # View a specific commit
git checkout <commit-hash> <file>  # Restore a file
```

### From Box:
- Go to https://app.box.com
- Navigate to the file
- Click "···" → "Version History"

## Tips

- **Commit often** - After each meaningful change
- **Sync to Box** - At end of work session or before turning off computer
- **Don't work directly in Box folder** - Always use the local folder
- **Jupyter autosaves** - But still manually save important work

## Files

- `.gitignore` - Tells Git which files to ignore
- `sync-to-box.ps1` - PowerShell sync script
- `sync-to-box.bat` - Quick launcher for sync script
