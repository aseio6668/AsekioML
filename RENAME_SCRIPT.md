# AsekioML Rename Script

## Quick PowerShell Commands to Complete the Rename

```powershell
# Fix the broken CMakeLists.txt target_link_libraries
(Get-Content CMakeLists.txt) -replace 'target_link_libraries\(TARGET_NAME asekioml\)', 'target_link_libraries($1 asekioml)' | Set-Content CMakeLists.txt

# Then fix specific instances with actual target names - you'll need to manually fix these:
# advanced_ml_features_demo, advanced_training_demo, model_serialization_demo, etc.
```

## Manual Fix Needed for CMakeLists.txt

The automated replacement broke the file. You need to manually fix all instances of:
`target_link_libraries(TARGET_NAME asekioml)`

And replace them with the correct target names like:
- `target_link_libraries(advanced_ml_features_demo asekioml)`
- `target_link_libraries(advanced_training_demo asekioml)`
- `target_link_libraries(model_serialization_demo asekioml)`
- etc.

## Alternative: Start Fresh

If the file is too corrupted, consider:
1. Backing up your current CMakeLists.txt
2. Starting with a clean version
3. Applying only the necessary renames systematically

The key changes needed are:
- Project name: AsekioML
- Library target: asekioml
- All target_link_libraries should reference asekioml instead of clmodel
