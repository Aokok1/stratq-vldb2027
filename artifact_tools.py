"""Build/verify the StratQ artifact and fetch only checksum-pinned inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parent
MANIFEST = "ARTIFACT_SHA256.json"
PREFIX = "stratq-artifact-vldb2027-rc1/"
STATIC = (
    "ARTIFACT.md", "ARTIFACT_RERUN.md", "artifact_tools.py", "requirements-artifact.txt",
    "PROOF_PACKAGE.md", "PROOF_AUDIT.md", "BASELINE_AUDIT.md", "BIBLIOGRAPHY_AUDIT.md",
    "NOVELTY_AUDIT.md", "VLDB_PC_REVIEW_SPRINT4.md", "SPRINT4_REPORT.md", "CMT_REGISTRATION_VLDB2027.md",
    "experiments/README.md", "figures/README.md",
)
EXCLUDE = {"results/artifact_independent_verification_sprint4.json"}
PATTERNS = (
    "src/stratq/*.py", "tests/test_*.py", "experiments/*.py", "results/*.json",
    "figures/*.pdf", "figures/*.png", "figures/figure_metrics.json", "figures/figure_metrics_sprint3.json",
    "4_LaTeX_Source/vldb_template/stratq*.tex",
    "4_LaTeX_Source/vldb_template/stratq_refs.bib",
    "4_LaTeX_Source/vldb_template/stratq.pdf",
    "4_LaTeX_Source/vldb_template/stratq.bbl",
    "4_LaTeX_Source/vldb_template/stratq.log",
    "4_LaTeX_Source/vldb_template/PAPER_CLAIM_AUDIT_SPRINT3.json",
    "4_LaTeX_Source/vldb_template/acmart.cls",
    "4_LaTeX_Source/vldb_template/pvldb.sty",
    "4_LaTeX_Source/vldb_template/ACM-Reference-Format.bst",
)


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build(output: Path):
    paths = {ROOT / path for path in STATIC}
    for pattern in PATTERNS:
        matches = set(ROOT.glob(pattern))
        if not matches:
            raise ValueError(f"Missing artifact inputs for {pattern}")
        paths |= matches
    contents = {}
    for path in sorted(paths):
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXCLUDE:
            continue
        if not path.is_file() or path.resolve() != path:
            raise ValueError(f"Unexpected file or path indirection: {path}")
        contents[rel] = path.read_bytes()
    manifest = {filename: hashlib.sha256(blob).hexdigest() for filename, blob in contents.items()}
    contents[MANIFEST] = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as zf:
        for name, blob in contents.items():
            zf.writestr(ZipInfo(PREFIX + name), blob, compress_type=ZIP_DEFLATED, compresslevel=9)
    with ZipFile(output) as zf:
        if zf.testzip() is not None or set(zf.namelist()) != {PREFIX + name for name in contents}:
            raise ValueError("Archive content verification failed")
        if any(hashlib.sha256(zf.read(PREFIX + name)).hexdigest() != digest
               for name, digest in manifest.items()):
            raise ValueError("Archive hash verification failed")
    print(f"{len(manifest)} payloads, {output.stat().st_size} bytes, ZIP SHA-256 {sha(output)}")


def verify():
    manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        path = ROOT / name
        if not path.resolve().is_relative_to(ROOT) or sha(path) != digest:
            raise ValueError(f"Artifact hash mismatch: {name}")
    print(f"Verified {len(manifest)} payload hashes.")


def display_path(target: Path):
    try:
        return target.relative_to(ROOT)
    except ValueError:
        return target


def download_verified(target: Path, urls, digest: str, expected_size: int):
    if target.exists():
        if target.stat().st_size != expected_size or sha(target) != digest:
            raise ValueError(f"Existing file has the wrong size/hash: {target}")
        print(f"Input verified: {display_path(target)}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    last_error = None
    for url in urls:
        try:
            request = Request(url, headers={"User-Agent": "StratQ-artifact/1.0"})
            h = hashlib.sha256()
            size = 0
            with urlopen(request, timeout=120) as response, part.open("wb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > expected_size:
                        raise ValueError(f"Download exceeds expected size: {url}")
                    h.update(block)
                    handle.write(block)
            if size != expected_size or h.hexdigest() != digest:
                raise ValueError(f"Download size/hash mismatch: {url}")
            os.replace(part, target)
            print(f"Input verified: {display_path(target)}")
            return
        except Exception as exc:  # try the checksum-equivalent mirror, if supplied
            last_error = exc
            part.unlink(missing_ok=True)
    raise RuntimeError(f"Could not fetch {target}: {last_error}")


def hf_urls(kind, repo, revision, path):
    primary = f"https://huggingface.co/{kind}/{repo}/resolve/{revision}/{path}"
    mirror = f"https://hf-mirror.com/{kind}/{repo}/resolve/{revision}/{path}"
    return [primary, mirror]


def fetch_data(core_only=False):
    def load(name):
        return json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))

    uci = "https://archive.ics.uci.edu/static/public/"
    fashion = load("fashion_live")
    gpu = load("sprint2_gpu_workloads")
    yelp = next(w for w in gpu["workloads"] if w["workload"] == "Yelp-DistilBERT-SST2")
    cifar = next(w for w in gpu["workloads"] if w["workload"] == "CIFAR10-ResNet34-GPU")
    items = [
        (ROOT / "data/adult_uci.zip", [uci + "2/adult.zip"], load("adult_count_replay")["archive_sha256"], None),
        (ROOT / "data/uci_sms.zip", [uci + "228/sms+spam+collection.zip"], load("sms_live")["archive_sha256"], None),
    ]
    for name in ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                 "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"):
        items.append((ROOT / "data/fashion" / name,
                      ["https://storage.googleapis.com/tensorflow/tf-keras-datasets/" + name],
                      fashion["sha256"][name], None))
    if not core_only:
        yelp_rev = "c1f9ee939b7d05667af864ee1cb066393154bf85"
        items.append((ROOT / "data/sprint2/yelp_test.parquet",
                      hf_urls("datasets", "Yelp/yelp_review_full", yelp_rev,
                              "yelp_review_full/test-00000-of-00001.parquet"),
                      yelp["dataset_sha256"], 23515519))
        items.append((ROOT / "data/sprint2/cifar-10-python.tar.gz",
                      ["https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"],
                      cifar["dataset_sha256"], 170498071))
        items.append((Path.home() / ".cache/torch/hub/checkpoints/resnet34-b627a593.pth",
                      ["https://download.pytorch.org/models/resnet34-b627a593.pth"],
                      cifar["backbone_sha256"], 87319819))
        model_rev = "714eb0fa89d2f80546fda750413ed43d93601a13"
        model_files = {
            "config.json": ("582122c8f414793d131e10022ce9ba04e3811a9da6389137ee2f18665b4f4d15", 629),
            "tokenizer_config.json": ("5ab9097b4149371c5fd52b2d6e26cb6f9c07c0d19fcfdda895b1adad6b57c3e0", 48),
            "vocab.txt": ("07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3", 231508),
            "model.safetensors": (yelp["model_sha256"], 267832558),
        }
        for name, (digest, size) in model_files.items():
            items.append((ROOT / "models/sprint2/distilbert_sst2" / name,
                          hf_urls("models", "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                                  model_rev, name), digest, size))
    for target, urls, digest, size in items:
        if size is None:
            # Core archives have a pinned hash but no predeclared size.
            if target.exists() and sha(target) == digest:
                print(f"Input verified: {display_path(target)}")
                continue
            size = 65_000_001
        download_verified(target, urls, digest, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("build").add_argument("zipfile", type=Path)
    sub.add_parser("verify")
    fetch = sub.add_parser("fetch-data")
    fetch.add_argument("--core-only", action="store_true")
    args = parser.parse_args()
    if args.action == "build":
        build(args.zipfile)
    elif args.action == "verify":
        verify()
    else:
        fetch_data(core_only=args.core_only)
