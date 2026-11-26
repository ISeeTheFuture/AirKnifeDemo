"""Build the Windows x64 demo from the project's simulation environment."""

import argparse
import json
import sys
import tarfile
from importlib.util import find_spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

source = Path(SPECPATH) / "source"
sys.path.insert(0, str(source))
from scan import CAPTURES, locate_tar

parser = argparse.ArgumentParser()
parser.add_argument("--tar", type=Path, help="Original vendor capture archive")
archive = parser.parse_args().tar or locate_tar()
if archive is None:
    raise SystemExit("Supply the capture archive after the spec: -- --tar <path>")

# Bundle only the fifteen CSVs read by load_scan. Keep the original data bytes
# and internal paths so the packaged and source versions use the same loader.
sample_tar = Path(workpath) / "Air Knife System Samples.tar"
with tarfile.open(archive, "r:*", encoding="utf-8") as incoming, tarfile.open(sample_tar, "w") as samples:
    members = {member.name.removeprefix("./"): member for member in incoming if member.isfile()}
    for sample, (capture, _, _) in CAPTURES.items():
        for axis in "XYZ":
            name = f"02 Proposal/{sample}/PCL_{capture}/PCL_{axis}_0.csv"
            member = members[name]
            entry = tarfile.TarInfo(name)
            entry.size = member.size
            with incoming.extractfile(member) as stream:
                samples.addfile(entry, stream)

robot = Path(find_spec("pybullet_data").origin).parent / "kuka_iiwa"
data = [
    (str(sample_tar), "."),
    (str(robot / "model.urdf"), "pybullet_data/kuka_iiwa"),
    (str(robot / "meshes"), "pybullet_data/kuka_iiwa/meshes"),
]
data += collect_data_files("open3d", includes=["resources/**"])
data += copy_metadata("dash")

# Preserve the notices supplied by the Python wheels and native runtime.
prefix = Path(sys.prefix)
data.append((str(prefix / "LICENSE_PYTHON.txt"), "licenses/python"))
for info in (prefix / "Lib/site-packages").glob("*.dist-info"):
    for notice in info.iterdir():
        if notice.name.lower().startswith(("license", "licence", "copying", "notice", "authors")):
            dest = Path("licenses") / info.name
            data.append((str(notice), str(dest / notice.name if notice.is_dir() else dest)))
for record in (prefix / "conda-meta").glob("*.json"):
    package = json.loads(record.read_text(encoding="utf-8"))
    cache = package.get("link", {}).get("source")
    if cache and (notices := Path(cache) / "info/licenses").is_dir():
        data.append((str(notices), f"licenses/conda/{package['name']}"))

a = Analysis(
    [str(source / "demo.py")],
    pathex=[str(source)],
    binaries=collect_dynamic_libs("open3d"),
    datas=data,
    hiddenimports=["open3d.cpu.pybind"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="AirKnifeDemo",
    console=False,
    upx=False,
)
