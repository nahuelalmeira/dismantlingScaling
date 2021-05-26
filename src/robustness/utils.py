from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
NETWORKS_DIR = ROOT_DIR / 'networks'
assert NETWORKS_DIR.is_dir()