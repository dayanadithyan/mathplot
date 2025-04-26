import os
import zipfile
from datetime import datetime

# derive from package
from mathplot import __version__

ADDON_NAME = "mathplot"
VERSION = __version__

def make_release():
    date_str = datetime.now().strftime("%Y%m%d")
    zip_name = f"{ADDON_NAME}-{VERSION}-{date_str}.zip"
    files = [
        # python modules
        os.path.join("mathplot", f) for f in [
            "__init__.py", "collections.py", "math_utils.py",
            "progress.py", "properties.py",
        ]
    ] + [
        # subpackages
        *[os.path.join("mathplot", d, f) for d in ["utils", "operators", "ui", "algorithms"] for f in os.listdir(os.path.join("mathplot", d)) if f.endswith('.py')]
    ]
    # optional assets (if any)

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            zf.write(file)
    print(f"created {zip_name} with {len(files)} files")

if __name__ == '__main__':
    make_release()