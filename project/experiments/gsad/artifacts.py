from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        converted = [_jsonable(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True)) if isinstance(
            value, (set, frozenset)
        ) else converted
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        return _jsonable(value.item())
    if isinstance(value, float) and not (value == value and abs(value) != float("inf")):
        raise ValueError("canonical JSON cannot contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_digest(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def write_canonical_json(path: Path, payload: Any) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(encoded).hexdigest()


def write_manifest(
    path: Path,
    inputs: Mapping[str, Any],
    config: Mapping[str, Any],
    split_audit: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "inputs": _jsonable(inputs),
        "config": _jsonable(config),
        "split_audit": _jsonable(split_audit),
    }
    payload = {**core, "manifest_digest": _payload_digest(core)}
    write_canonical_json(path, payload)
    return payload


@dataclass(frozen=True)
class FreezeToken:
    digest: str
    config_digest: str
    gate_digest: str
    token_path: Path


def _primary_passed(development_gates: Mapping[str, Any]) -> bool:
    primary = development_gates.get("PRIMARY")
    if primary is None:
        return False
    if isinstance(primary, Mapping):
        return bool(primary.get("passed", False))
    return bool(getattr(primary, "passed", False))


def freeze_candidate(
    config: Mapping[str, Any],
    development_gates: Mapping[str, Any],
    path: Path,
) -> FreezeToken:
    if not _primary_passed(development_gates):
        raise ValueError("candidate cannot be frozen unless PRIMARY gate passed")
    config_digest = _payload_digest(config)
    gate_digest = _payload_digest(development_gates)
    core = {"config_digest": config_digest, "gate_digest": gate_digest}
    digest = _payload_digest(core)
    payload = {**core, "digest": digest}
    write_canonical_json(path, payload)
    return FreezeToken(
        digest=digest,
        config_digest=config_digest,
        gate_digest=gate_digest,
        token_path=Path(path),
    )


def _validate_token(token: FreezeToken) -> None:
    expected = _payload_digest(
        {"config_digest": token.config_digest, "gate_digest": token.gate_digest}
    )
    if token.digest != expected:
        raise ValueError("freeze token digest is invalid")
    try:
        stored = json.loads(Path(token.token_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("freeze token file is missing or invalid") from exc
    if stored.get("digest") != token.digest:
        raise ValueError("freeze token file digest does not match token")


def claim_locked_run(token: FreezeToken, results_dir: Path) -> Path:
    _validate_token(token)
    destination = Path(results_dir)
    global_claim_path = token.token_path.parent / "LOCKED_EVALUATION_CLAIMED.json"
    payload = {
        "freeze_digest": token.digest,
        "config_digest": token.config_digest,
        "gate_digest": token.gate_digest,
    }
    try:
        with global_claim_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_bytes(payload).decode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise FileExistsError(
            f"locked evaluation already claimed: {global_claim_path}"
        ) from exc
    destination.mkdir(parents=True, exist_ok=True)
    result_claim_path = destination / "LOCKED_EVALUATION_CLAIMED.json"
    if result_claim_path.resolve() != global_claim_path.resolve():
        write_canonical_json(result_claim_path, payload)
    return global_claim_path
