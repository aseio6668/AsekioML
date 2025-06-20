# AsekioML Rename Plan

## Files and Changes Completed ✅
- [x] docs/STATISTICAL_MONITORING.md - Updated project references
- [x] README.md - Updated title and description
- [x] CMakeLists.txt - Updated project name and library target
- [x] include/clmodel.hpp - Updated header comments

## Still Need to Change

### 1. Directory Structure
- Rename root folder: `CLModel` → `AsekioML`
- Consider renaming: `include/clmodel.hpp` → `include/asekioml.hpp`

### 2. Namespace Changes
**Current:** `namespace clmodel {`
**Target:** `namespace asekioml {`

**Files to Update:**
- All header files in `include/` directory
- All source files in `src/` directory
- All example files in `examples/` directory

### 3. CMakeLists.txt Updates
- [x] Project name: `CLModel` → `AsekioML`
- [x] Library target: `clmodel` → `asekioml`
- [ ] Update all `target_link_libraries(xxx clmodel)` → `target_link_libraries(xxx asekioml)`
- [ ] Update include path references

### 4. Include Guards and Preprocessor Definitions
- Change `CLMODEL_OPENMP_SUPPORT` → `ASEKIOML_OPENMP_SUPPORT`
- Update any other `CLMODEL_*` definitions

### 5. Documentation Updates
- Update all remaining references to CLModel in docs/
- Update code examples and usage instructions
- Update build instructions

## Search and Replace Strategy

### Global Find/Replace Needed:
1. `clmodel` → `asekioml` (namespace, library names)
2. `CLModel` → `AsekioML` (project references)
3. `CLMODEL` → `ASEKIOML` (preprocessor definitions)

### Files to Check:
- All `.hpp` files in `include/`
- All `.cpp` files in `src/`
- All `.cpp` files in `examples/`
- All `.md` files in `docs/`
- `CMakeLists.txt`
- Any configuration files
