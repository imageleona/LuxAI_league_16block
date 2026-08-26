"""Build a standalone Lux AI 2021 submission without deleting or overwriting files.

The league's ``final_agent`` directory intentionally contains only the model
weights and two configuration files.  Kaggle's Lux runner additionally needs
``main.py`` and the Python modules used for observation processing, model
construction, and action post-processing.  This script combines the evaluated
agent with the runtime from a known working submission and writes both a folder
and, optionally, a ZIP whose root contains ``main.py``.

The destination folder and ZIP must not already exist.  This is deliberate: a
packaging mistake must never erase a previous model or submission artifact.
"""

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = (
    REPO_ROOT
    / "internal_testing"
    / "hall_of_fame"
    / "11-24_12-56-23_062179520_must_research"
)
RL_AGENT_DIR = Path("lux_ai") / "rl_agent"
MODEL_CONFIG = RL_AGENT_DIR / "config.yaml"
AGENT_CONFIG = RL_AGENT_DIR / "rl_agent_config.yaml"
DEPLOYMENT_WRAPPER = RL_AGENT_DIR / "rl_agent.py"
CURRENT_RUNTIME_DIRS = (
    Path("lux"),
    Path("lux_gym"),
    Path("nns"),
    Path("rl_agent"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _agent_files(agent_dir: Path) -> Tuple[Path, Path, Path]:
    rl_dir = agent_dir / RL_AGENT_DIR
    if not rl_dir.is_dir():
        raise FileNotFoundError("Agent runtime directory not found: {}".format(rl_dir))
    weights = sorted(rl_dir.glob("*.pt"))
    if len(weights) != 1:
        raise ValueError(
            "Expected exactly one checkpoint in {}, found {}".format(rl_dir, len(weights))
        )
    model_config = rl_dir / "config.yaml"
    agent_config = rl_dir / "rl_agent_config.yaml"
    for required in (model_config, agent_config):
        if not required.is_file():
            raise FileNotFoundError("Required agent configuration not found: {}".format(required))
    return weights[0], model_config, agent_config


def _template_files(template_dir: Path) -> Iterable[Tuple[Path, Path]]:
    """Yield safe template files as ``(source, relative_path)`` pairs."""
    template_dir = template_dir.resolve()
    for source in sorted(path for path in template_dir.rglob("*") if path.is_file()):
        relative = source.relative_to(template_dir)
        if "__pycache__" in relative.parts or source.suffix in {".pyc", ".pyo"}:
            continue
        # Never copy the template's model or model-specific configurations.
        if source.suffix == ".pt" or relative in {
                MODEL_CONFIG, AGENT_CONFIG, DEPLOYMENT_WRAPPER}:
            continue
        yield source, relative


def _current_runtime_files() -> Iterable[Tuple[Path, Path]]:
    """Yield the inference runtime that matches the trained checkpoint.

    The known submission supplies the Kaggle entry point and general folder
    layout.  Model-building code must come from the current project, however:
    Stage 1 uses adaptive embedding widths which the older template predates.
    """
    current_lux_ai = REPO_ROOT / "lux_ai"
    candidates = [
        current_lux_ai / "__init__.py",
        current_lux_ai / "utility_constants.py",
        current_lux_ai / "utils.py",
    ]
    for runtime_dir in CURRENT_RUNTIME_DIRS:
        candidates.extend((current_lux_ai / runtime_dir).rglob("*"))

    for source in sorted(path for path in candidates if path.is_file()):
        relative_under_lux = source.relative_to(current_lux_ai)
        relative = Path("lux_ai") / relative_under_lux
        if "__pycache__" in relative.parts or source.suffix in {".pyc", ".pyo"}:
            continue
        if relative in {MODEL_CONFIG, AGENT_CONFIG} or source.suffix == ".pt":
            continue
        yield source, relative


def _write_zip(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        raise FileExistsError("Refusing to overwrite existing archive: {}".format(zip_path))
    with zipfile.ZipFile(str(zip_path), mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(path for path in folder.rglob("*") if path.is_file()):
            archive.write(str(source), str(source.relative_to(folder)).replace("\\", "/"))


def build_submission(
        agent_dir: Path,
        template_dir: Path,
        output_dir: Path,
        create_zip: bool = True,
) -> Dict[str, object]:
    agent_dir = agent_dir.expanduser().resolve()
    template_dir = template_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite existing submission: {}".format(output_dir))
    if not template_dir.is_dir():
        raise FileNotFoundError("Submission template not found: {}".format(template_dir))

    weights, model_config, agent_config = _agent_files(agent_dir)
    zip_path = output_dir.with_suffix(".zip")
    if create_zip and zip_path.exists():
        raise FileExistsError("Refusing to overwrite existing archive: {}".format(zip_path))

    copied = set()
    for source, relative in _template_files(template_dir):
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(destination))
        copied.add(relative.as_posix())

    # Overlay the current inference runtime so model construction exactly
    # matches the code used to train the supplied checkpoint.  In particular,
    # the Stage 1 weights use adaptive embedding widths that are not understood
    # by the older, otherwise proven submission template.
    for source, relative in _current_runtime_files():
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(destination))
        copied.add(relative.as_posix())

    output_rl_dir = output_dir / RL_AGENT_DIR
    output_weights = output_rl_dir / "stage1_final_weights.pt"
    shutil.copy2(str(weights), str(output_weights))
    shutil.copy2(str(model_config), str(output_rl_dir / "config.yaml"))
    shutil.copy2(str(agent_config), str(output_rl_dir / "rl_agent_config.yaml"))

    packaged_weights = sorted(output_rl_dir.glob("*.pt"))
    if len(packaged_weights) != 1:
        raise RuntimeError(
            "Packaged submission must contain exactly one checkpoint; found {}".format(
                len(packaged_weights)
            )
        )
    main_path = output_dir / "main.py"
    if not main_path.is_file():
        raise RuntimeError("Packaged submission has no root main.py")

    manifest = {
        "format": "lux-ai-2021-python-submission",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_agent": str(agent_dir),
        "template": str(template_dir),
        "weights": str(output_weights.relative_to(output_dir)).replace("\\", "/"),
        "weights_bytes": output_weights.stat().st_size,
        "weights_sha256": _sha256(output_weights),
        "runtime_files_copied": len(copied),
    }
    manifest_path = output_dir / "submission_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if create_zip:
        _write_zip(output_dir, zip_path)
        manifest["zip"] = str(zip_path)
        manifest["zip_bytes"] = zip_path.stat().st_size
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, type=Path,
                        help="Evaluated AgentSpec folder containing lux_ai/rl_agent")
    parser.add_argument("--output", required=True, type=Path,
                        help="New submission directory; it must not already exist")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                        help="Known working standalone submission used as the runtime template")
    parser.add_argument("--no-zip", action="store_true",
                        help="Create only the folder, not the sibling submission ZIP")
    args = parser.parse_args()
    result = build_submission(
        agent_dir=args.agent,
        template_dir=args.template,
        output_dir=args.output,
        create_zip=not args.no_zip,
    )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
