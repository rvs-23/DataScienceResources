"""Image Mover

This script finds image files recursively in a
folder structure and moves them to an `images/`
folder. It creates the `images/` folder if it
doesn't already exist.

You can define which image files to move by
changing the `IMAGE_EXTENSIONS` constant.
"""

from pathlib import Path

FULL_PATH = Path.home() / "python-basics-exercises" / "practice_files"
IMAGE_EXTENSIONS = (".png", ".gif", ".jpg")


images_dir = FULL_PATH / "images"
images_dir.mkdir(exist_ok=True)

for path in FULL_PATH.rglob("*"):
    if path.suffix in IMAGE_EXTENSIONS:
        path.replace(images_dir / path.name)
