import json, hashlib, urllib.request, sys

ref = sys.argv[1]         # e.g., a tag, branch, or commit SHA
version = sys.argv[2]     # e.g., "3.15.1-vcpkg.1"

# Construct GitHub tarball URL for the given ref
tarball_url = f"https://github.com/ericwait/hydra-image-processor/archive/{ref}.tar.gz"

# Download the tarball
print(f"Downloading tarball from: {tarball_url}")
tarball_data = urllib.request.urlopen(tarball_url).read()

# Compute SHA512
sha512 = hashlib.sha512(tarball_data).hexdigest()
print(f"Computed SHA512: {sha512}")

# Patch vcpkg.json
with open("vcpkg.json", "r") as f:
    vcpkg = json.load(f)

vcpkg["version"] = version

with open("vcpkg.json", "w") as f:
    json.dump(vcpkg, f, indent=2)
    f.write("\n")

# Patch portfile.cmake
portfile_path = "ports/hydra/portfile.cmake"
with open(portfile_path, "r") as f:
    lines = f.readlines()

with open(portfile_path, "w") as f:
    for line in lines:
        if line.strip().startswith("REF "):
            f.write(f'    REF {ref}\n')
        elif line.strip().startswith("SHA512 "):
            f.write(f'    SHA512 {sha512}\n')
        else:
            f.write(line)
