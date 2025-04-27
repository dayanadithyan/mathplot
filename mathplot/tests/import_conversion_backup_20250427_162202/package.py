import os
import zipfile
from datetime import datetime

# derive version from the package
from mathplot import __version__ as VERSION

ADDON_NAME = "mathplot"


def make_release():
    """
    creates a zip file of the mathplot add-on under the name:
      mathplot-<version>-<YYYYMMDD>.zip
    """
    date_str = datetime.now().strftime("%Y%m%d")
    zip_name = f"{ADDON_NAME}-{VERSION}-{date_str}.zip"

    # collect all .py files under mathplot/
    files = []
    for root, _, filenames in os.walk("mathplot"):
        for fn in filenames:
            if fn.endswith(".py"):
                path = os.path.join(root, fn)
                files.append(path)

    # optionally include README and LICENSE if desired
    # files += ["README.md", "LICENSE"]

    print(f"packing {len(files)} files into {zip_name}...")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            zf.write(file_path)
    print(f"created {zip_name}")


if __name__ == "__main__":
    make_release()
