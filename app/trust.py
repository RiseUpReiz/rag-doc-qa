"""Source trust: which corpus documents are approved ("official") versus "unverified".

A manifest lists approved files by name with the SHA-256 of their content:

    {"official": {"travel_policy.md": "<sha256 hex>", ...}}

A document is official only if its filename is listed AND its hash matches, so a renamed
or edited copy of an approved file is unverified. Anything not listed is unverified too.

Create a manifest with:
    python -m app.trust make-manifest <corpus_dir> <file> [<file> ...] --out <path>
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

OFFICIAL, UNVERIFIED = "official", "unverified"
TRUST_LEVELS = (OFFICIAL, UNVERIFIED)

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class ManifestError(ValueError):
    """The trust manifest is missing or malformed."""


def hash_text(text: str) -> str:
    """SHA-256 of text after stripping a BOM and normalising line endings to \\n.

    This makes the same document hash identically whether it was saved on Windows or Linux.
    """
    text = text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    """Hash a document file. Text files are hashed as normalised text, PDFs as raw bytes."""
    if path.suffix.lower() == ".pdf":
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return hash_text(path.read_bytes().decode("utf-8"))


def load_manifest(path: Path) -> dict[str, str]:
    """Read and validate a manifest, returning {filename: sha256}. Raises ManifestError if unusable."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestError(f"Trust manifest not found: {path}") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Trust manifest {path} could not be read: {exc}") from exc

    official = data.get("official") if isinstance(data, dict) else None
    if not isinstance(official, dict):
        raise ManifestError(f'Trust manifest {path} must be a JSON object with an "official" object')
    for name, digest in official.items():
        if Path(name).name != name or not name:
            raise ManifestError(f"Trust manifest {path}: {name!r} must be a bare filename")
        if not isinstance(digest, str) or not _SHA256_HEX.fullmatch(digest):
            raise ManifestError(f"Trust manifest {path}: hash for {name!r} must be 64 lowercase hex characters")
    return dict(official)


def load_manifest_if_needed(path: Path, required: bool) -> dict[str, str] | None:
    """Load the manifest; a missing file is an error only when required, otherwise None."""
    if not required and not Path(path).exists():
        return None
    return load_manifest(path)


def trust_level(path: Path, manifest: dict[str, str]) -> str:
    """official if the file's name is in the manifest and its content hash matches, else unverified."""
    expected = manifest.get(path.name)
    return OFFICIAL if expected is not None and hash_file(path) == expected else UNVERIFIED


def make_manifest(corpus_dir: Path, filenames: list[str]) -> dict:
    """Build a manifest marking the named files in corpus_dir as official."""
    official = {}
    for name in filenames:
        path = Path(corpus_dir) / name
        if Path(name).name != name:
            raise ValueError(f"{name!r} must be a filename inside {corpus_dir}, not a path")
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")
        official[name] = hash_file(path)
    return {"official": dict(sorted(official.items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make-manifest", help="hash approved files and write a manifest")
    make.add_argument("corpus_dir", type=Path)
    make.add_argument("files", nargs="+", help="filenames in corpus_dir to mark as official")
    make.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        manifest = make_manifest(args.corpus_dir, args.files)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest['official'])} official file(s) to {args.out}")


if __name__ == "__main__":
    main()
