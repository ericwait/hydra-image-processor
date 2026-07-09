#!/usr/bin/env python3
"""
Download GitHub Actions artifacts for local testing.

Usage:
    python scripts/download_artifacts.py --run-id <RUN_ID>
"""

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, urlopen


def get_artifacts(repo, run_id, token=None):
    """Get list of artifacts from a workflow run."""
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts"

    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"

    req = urlopen(url)
    data = json.loads(req.read())
    return data["artifacts"]


def download_artifact(artifact, output_dir, token=None):
    """Download a single artifact."""
    name = artifact["name"]
    url = artifact["archive_download_url"]

    output_path = Path(output_dir) / f"{name}.zip"

    print(f"Downloading {name}...")

    # Use gh CLI if available for authenticated downloads
    if token or subprocess.run(["gh", "--version"], capture_output=True).returncode == 0:
        subprocess.run([
            "gh", "api", url,
            "-H", "Accept: application/vnd.github+json",
            "-H", "X-GitHub-Api-Version: 2022-11-28",
            "--output", str(output_path)
        ])
    else:
        urlretrieve(url, output_path)

    # Extract
    extract_dir = Path(output_dir) / name
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(output_path, 'r') as zf:
        zf.extractall(extract_dir)

    output_path.unlink()  # Delete zip file
    print(f"  Extracted to {extract_dir}")

    return extract_dir


def main():
    parser = argparse.ArgumentParser(description="Download GitHub Actions artifacts")
    parser.add_argument("--repo", default="ericwait/hydra-image-processor",
                        help="GitHub repository (owner/name)")
    parser.add_argument("--run-id", required=True,
                        help="Workflow run ID")
    parser.add_argument("--output-dir", default="./downloaded_artifacts",
                        help="Output directory for artifacts")
    parser.add_argument("--token",
                        help="GitHub personal access token (optional)")
    parser.add_argument("--filter",
                        help="Filter artifacts by name pattern")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get artifacts
    artifacts = get_artifacts(args.repo, args.run_id, args.token)

    if not artifacts:
        print("No artifacts found for this run")
        return 1

    print(f"Found {len(artifacts)} artifacts:")
    for a in artifacts:
        print(f"  - {a['name']}")

    # Filter if requested
    if args.filter:
        artifacts = [a for a in artifacts if args.filter in a["name"]]
        print(f"\nFiltered to {len(artifacts)} artifacts")

    # Download artifacts
    print("\nDownloading artifacts...")
    for artifact in artifacts:
        download_artifact(artifact, output_dir, args.token)

    print(f"\nAll artifacts downloaded to {output_dir}")

    # Create test script
    test_script = output_dir / "test_import.py"
    test_script.write_text("""
import sys
import os
from pathlib import Path

# Add each artifact directory to path and try to import
artifact_dirs = [d for d in Path('.').iterdir() if d.is_dir()]

for dir in artifact_dirs:
    print(f"\\nTesting {dir.name}...")
    sys.path.insert(0, str(dir))

    try:
        import Hydra
        print(f"  ✓ Successfully imported Hydra from {dir.name}")

        # Try to call a basic function (will fail if no CUDA)
        try:
            # info = Hydra.DeviceCount()
            print(f"  ✓ Module functional")
        except Exception as e:
            print(f"  ⚠ Module imported but CUDA not available: {e}")

        del sys.modules['Hydra']
    except ImportError as e:
        print(f"  ✗ Failed to import: {e}")
    finally:
        sys.path.pop(0)
""")

    print(f"\nCreated test script: {test_script}")
    print("Run it with: python downloaded_artifacts/test_import.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())