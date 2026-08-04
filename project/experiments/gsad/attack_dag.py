from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class StructuredSet:
    nodes: frozenset[str]
    descendants: frozenset[str]
    leaf_equivalent_size: int
    objective: float
    coverage_preserved: bool


class AttackDAG:
    """A tactic-to-parent-technique multi-parent DAG."""

    def __init__(
        self,
        tactic_to_techniques: Mapping[str, Iterable[str]],
        vocab: Sequence[str] | None = None,
    ):
        cleaned = {
            str(tactic): frozenset(str(label) for label in labels)
            for tactic, labels in tactic_to_techniques.items()
            if str(tactic)
        }
        self.tactic_to_techniques = {
            tactic: labels for tactic, labels in cleaned.items() if labels
        }
        self.tactic_ids = tuple(sorted(self.tactic_to_techniques))
        mapped = frozenset().union(*self.tactic_to_techniques.values()) if cleaned else frozenset()
        self.vocab = tuple(str(label) for label in (vocab if vocab is not None else sorted(mapped)))
        if len(set(self.vocab)) != len(self.vocab):
            raise ValueError("ATT&CK vocabulary must be unique")
        self._technique_to_tactics: dict[str, frozenset[str]] = {}
        for label in self.vocab:
            self._technique_to_tactics[label] = frozenset(
                tactic
                for tactic, descendants in self.tactic_to_techniques.items()
                if label in descendants
            )

        known = sorted(mapped)
        self._known_techniques = tuple(known)
        self._technique_index = {label: index for index, label in enumerate(known)}
        self._tactic_masks = tuple(
            sum(1 << self._technique_index[label] for label in self.tactic_to_techniques[tactic])
            for tactic in self.tactic_ids
        )
        self._subset_descendant_masks = self._build_subset_descendant_masks()
        missing = [label for label in self.vocab if not self._technique_to_tactics[label]]
        self.mapping_audit = {
            "vocab_size": len(self.vocab),
            "mapped_techniques": len(self.vocab) - len(missing),
            "missing_techniques": len(missing),
            "missing_ids": missing,
            "tactic_count": len(self.tactic_ids),
            "multi_parent_techniques": sum(
                len(self._technique_to_tactics[label]) > 1 for label in self.vocab
            ),
        }

    @classmethod
    def from_edges(cls, edges: Mapping[str, Iterable[str]]) -> "AttackDAG":
        vocab = sorted({str(label) for labels in edges.values() for label in labels})
        return cls(edges, vocab=vocab)

    @classmethod
    def from_stix(cls, path: Path, vocab: Sequence[str]) -> "AttackDAG":
        with Path(path).open("r", encoding="utf-8") as handle:
            bundle = json.load(handle)
        vocab_set = {str(label) for label in vocab}
        edges: dict[str, set[str]] = {}
        for obj in bundle.get("objects", []):
            if obj.get("type") != "attack-pattern":
                continue
            if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
                continue
            external_id = None
            for reference in obj.get("external_references", []):
                candidate = str(reference.get("external_id", ""))
                if reference.get("source_name") == "mitre-attack" and candidate.startswith("T"):
                    external_id = candidate
                    break
            if external_id is None:
                continue
            parent_id = external_id.split(".", 1)[0]
            if parent_id not in vocab_set:
                continue
            for phase in obj.get("kill_chain_phases", []):
                if phase.get("kill_chain_name") != "mitre-attack":
                    continue
                tactic = str(phase.get("phase_name", "")).strip()
                if tactic:
                    edges.setdefault(tactic, set()).add(parent_id)
        return cls(edges, vocab=vocab)

    def _build_subset_descendant_masks(self) -> tuple[int, ...]:
        masks = [0] * (1 << len(self.tactic_ids))
        for subset in range(1, len(masks)):
            least_bit = subset & -subset
            tactic_index = least_bit.bit_length() - 1
            masks[subset] = masks[subset ^ least_bit] | self._tactic_masks[tactic_index]
        return tuple(masks)

    def tactics_for(self, technique_id: str) -> frozenset[str]:
        return self._technique_to_tactics.get(str(technique_id), frozenset())

    def descendants_of_tactics(self, tactics: Iterable[str]) -> frozenset[str]:
        descendants: set[str] = set()
        for tactic in tactics:
            descendants.update(self.tactic_to_techniques.get(str(tactic), frozenset()))
        return frozenset(descendants)

    def descendants(self, nodes: Iterable[str]) -> frozenset[str]:
        descendants: set[str] = set()
        for raw_node in nodes:
            node = str(raw_node)
            if node in self.tactic_to_techniques:
                descendants.update(self.tactic_to_techniques[node])
            else:
                descendants.add(node)
        return frozenset(descendants)

    def known_mask(self, labels: Iterable[str]) -> tuple[int, frozenset[str]]:
        mask = 0
        unknown: set[str] = set()
        for raw_label in labels:
            label = str(raw_label)
            index = self._technique_index.get(label)
            if index is None:
                unknown.add(label)
            else:
                mask |= 1 << index
        return mask, frozenset(unknown)

    def labels_from_mask(self, mask: int) -> frozenset[str]:
        labels: set[str] = set()
        remaining = int(mask)
        while remaining:
            least_bit = remaining & -remaining
            index = least_bit.bit_length() - 1
            labels.add(self._known_techniques[index])
            remaining ^= least_bit
        return frozenset(labels)


def compress_leaf_set(
    gamma: frozenset[str],
    dag: AttackDAG,
    lam: float,
    max_nodes: int,
) -> StructuredSet:
    """Exactly minimize leaf-equivalent size plus display-node penalty."""

    if lam < 0:
        raise ValueError("lam must be nonnegative")
    if max_nodes < 0:
        raise ValueError("max_nodes must be nonnegative")
    gamma = frozenset(str(label) for label in gamma)
    if not gamma:
        return StructuredSet(
            nodes=frozenset(),
            descendants=frozenset(),
            leaf_equivalent_size=0,
            objective=0.0,
            coverage_preserved=True,
        )

    gamma_mask, unknown_gamma = dag.known_mask(gamma)
    best_key: tuple[float, int, int, tuple[str, ...]] | None = None
    best_nodes: frozenset[str] | None = None
    best_descendant_mask: int | None = None
    full_known_mask = (1 << len(dag._known_techniques)) - 1

    for tactic_subset, covered_mask in enumerate(dag._subset_descendant_masks):
        uncovered_mask = gamma_mask & (full_known_mask ^ covered_mask)
        node_count = tactic_subset.bit_count() + uncovered_mask.bit_count() + len(unknown_gamma)
        if node_count > max_nodes:
            continue
        descendant_mask = covered_mask | uncovered_mask
        leaf_size = descendant_mask.bit_count() + len(unknown_gamma)
        numeric_key = (
            float(leaf_size + lam * node_count),
            leaf_size,
            node_count,
        )
        if best_key is not None and numeric_key > best_key[:3]:
            continue
        tactics = {
            dag.tactic_ids[index]
            for index in range(len(dag.tactic_ids))
            if tactic_subset & (1 << index)
        }
        uncovered_labels = set(dag.labels_from_mask(uncovered_mask)) | set(unknown_gamma)
        nodes = frozenset(tactics | uncovered_labels)
        key = (*numeric_key, tuple(sorted(nodes)))
        if best_key is None or key < best_key:
            best_key = key
            best_nodes = nodes
            best_descendant_mask = descendant_mask

    if best_key is None or best_nodes is None or best_descendant_mask is None:
        raise ValueError("no feasible structured representation under max_nodes")
    best_descendants = dag.labels_from_mask(best_descendant_mask) | unknown_gamma
    coverage_preserved = gamma.issubset(best_descendants)
    if not coverage_preserved:
        raise AssertionError("DAG compression dropped a leaf from the prediction set")
    return StructuredSet(
        nodes=best_nodes,
        descendants=best_descendants,
        leaf_equivalent_size=len(best_descendants),
        objective=best_key[0],
        coverage_preserved=True,
    )
