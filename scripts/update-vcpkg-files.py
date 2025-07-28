import json, hashlib, urllib.request, sys

tag = sys.argv[1]
version = sys.argv[2]

tarball_url = f"https://github.com/ericwait/hydra-image-processor/archive/{tag}.tar.gz"
tarball_data = urllib.request.urlopen(tarball_url).read()
sha512 = hashlib.sha512(tarball_data).hexdigest()

# Update vcpkg.json
with open("vcpkg.json") as f:
    data = json.load(f)
data["version"] = version
with open("vcpkg.json", "w") as f:
    json.dump(data, f, indent=2)

# Update portfile.cmake
lines = []
with open("ports/hydra/portfile.cmake") as f:
    for line in f:
        if "REF " in line:
            line = f"    REF {tag}\n"
        elif "SHA512" in line:
            line = f"    SHA512 {sha512}\n"
        lines.append(line)
with open("ports/hydra/portfile.cmake", "w") as f:
    f.writelines(lines)
