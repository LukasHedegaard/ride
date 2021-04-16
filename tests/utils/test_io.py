import json
import shutil
from pathlib import Path

from ride.utils import io


def test_bump_version():
    temp_dir = Path(__file__).parent / "temp"
    temp_dir.mkdir(exist_ok=True, parents=True)

    def write_data(path: Path):
        with path.open("w", encoding="utf-8") as f:
            json.dump({"dummy": 42}, f)

    p1 = temp_dir / "dummy_file.json"
    p2 = temp_dir / "dummy_file_1.json"
    p3 = temp_dir / "dummy_file_2.json"

    # Ensure files don't exist beforehand
    try:
        p1.unlink()
        p2.unlink()
        p3.unlink()
    except FileNotFoundError:
        pass

    # Non existing file
    assert io.bump_version(p1) == p1
    write_data(p1)
    assert p1.exists()

    # File exists (with no '_x' prepended)
    assert not p2.exists()
    assert io.bump_version(p1) == p2
    write_data(p2)
    assert p2.exists()

    # File exists (with '_x' prepended)
    assert not p3.exists()
    assert io.bump_version(p1) == p3
    assert io.bump_version(p2) == p3

    # Clean up
    shutil.rmtree(str(temp_dir))
