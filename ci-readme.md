# CI/CD Setup for Hydra Image Processor

This document describes how to build and distribute the Hydra Python extensions using GitHub Actions.

## Files to Add to Your Repository

1. **`.github/workflows/build-python-artifacts.yml`** - Main CI workflow that builds the Python extensions
2. **`scripts/download_artifacts.py`** - Helper script to download artifacts from GitHub Actions
3. **`test_hydra_module.py`** - Test script to verify the built module works correctly
4. **Update `CMakeLists.txt`** - Change `HYDRA_MODULE_NAME` from "HIP" to "Hydra"

## Build Process

### Step 1: Update Your CMakeLists.txt

Change the module name in your root `CMakeLists.txt`:
```cmake
set(HYDRA_MODULE_NAME "Hydra")  # Changed from "HIP"
```

### Step 2: Commit and Push the Workflow

```bash
# Create the workflow directory
mkdir -p .github/workflows

# Copy the workflow file
cp build-python-artifacts.yml .github/workflows/

# Create scripts directory
mkdir -p scripts
cp download_artifacts.py scripts/

# Commit and push
git add .github/workflows/build-python-artifacts.yml
git add scripts/download_artifacts.py
git add test_hydra_module.py
git commit -m "Add CI/CD for Python extensions with Hydra module name"
git push
```

### Step 3: Monitor the Build

1. Go to your GitHub repository
2. Click on the "Actions" tab
3. Watch the workflow run
4. Builds will create artifacts for:
   - Linux (`.so` files) with CUDA 12.2 and 12.3
   - Windows (`.pyd` files) with CUDA 12.2 and 12.3
   - Python versions 3.9, 3.10, and 3.11

## Downloading and Testing Artifacts

### Method 1: Via GitHub Web Interface

1. Go to Actions tab
2. Click on a completed workflow run
3. Scroll down to "Artifacts"
4. Download the artifacts you need

### Method 2: Using the Download Script

```bash
# Find your run ID from the GitHub Actions page
python scripts/download_artifacts.py --run-id <RUN_ID>

# The script will download all artifacts and create a test script
cd downloaded_artifacts
python test_import.py
```

### Method 3: Using GitHub CLI

```bash
# List recent workflow runs
gh run list --workflow=build-python-artifacts.yml

# Download all artifacts from a specific run
gh run download <RUN_ID>
```

## Testing the Module

### Quick Test
```python
import sys
sys.path.insert(0, '/path/to/artifact')
import Hydra
print(Hydra.Info())
```

### Comprehensive Test
```bash
python test_hydra_module.py
```

## Using in Your Project

### For [[home-media-ai]] or Other Projects

1. Download the appropriate artifact for your platform:
   - Linux: `Hydra-linux-cuda12.2-py3.10/Hydra.so`
   - Windows: `Hydra-windows-cuda1220-py3.10/Hydra.pyd`

2. Copy to your project:
```bash
# Linux
cp downloaded_artifacts/Hydra-linux-cuda12.2-py3.10/Hydra.so ~/home-media-ai/

# Windows
copy downloaded_artifacts\Hydra-windows-cuda1220-py3.10\Hydra.pyd C:\projects\home-media-ai\
```

3. Import in Python:
```python
import Hydra

# Check available functions
info = Hydra.Info()
for cmd in info:
    print(cmd.command)

# Use the module
import numpy as np
image = np.random.rand(512, 512).astype(np.float32)
filtered = Hydra.Gaussian(image, sigma=[2.0, 2.0])
```

## Artifact Naming Convention

Artifacts are named with the pattern:
```
Hydra-{platform}-cuda{version}-py{python_version}
```

Examples:
- `Hydra-linux-cuda12.2-py3.10`
- `Hydra-windows-cuda1220-py3.11`

## Troubleshooting

### Import Errors

If you get import errors, check:
1. **CUDA Runtime**: Ensure CUDA 12.x is installed
2. **Python Version**: Match the Python version of the artifact
3. **Dependencies**: Install numpy: `pip install numpy`
4. **Library Path**: On Linux, you may need to set `LD_LIBRARY_PATH`

### CUDA Not Found

The module will import but operations may fail if CUDA is not available.
This is normal for testing on systems without GPUs.

### Windows Specific

On Windows, you may need:
- Visual C++ Redistributables
- CUDA Toolkit 12.x
- Ensure `.pyd` file is in a directory on your Python path

## Next Steps

Once testing is successful:

1. **Create Wheels**: Package as `.whl` files for pip installation
2. **Conda Package**: Create conda recipe for conda-forge submission
3. **Release Automation**: Auto-publish on GitHub releases

## Conda-forge Submission (Future)

After successful testing, we'll:
1. Fork conda-forge/staged-recipes
2. Add recipe in `recipes/hydra-image-processor/`
3. Submit PR for review
4. Once accepted, get a feedstock repository
5. Automatic builds on each release

## Support

For issues or questions:
- Check the [Actions logs](https://github.com/ericwait/hydra-image-processor/actions)
- Open an issue on the repository
- Check the [Hydra website](https://www.hydraimageprocessor.com)