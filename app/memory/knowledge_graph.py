"""Directed Acyclic Graph (DAG) Learner Knowledge Graph.

Models the learner's mastery across foundational and frontier AI concepts as a
verified Directed Acyclic Graph (DAG) persisted in Google Cloud Firestore.
Enforces topological acyclicity, tracks mastery tiers (Novice -> Practitioner ->
Hero), computes 3D cosine-similarity vectors for skill-tree visualization, and
schedules non-blocking async Firestore updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from app.memory.firestore_client import (
    FirestoreSessionStore,
    compute_concept_embedding,
    get_firestore_store,
)
from app.observability.logging import get_logger

logger = get_logger("memory.knowledge_graph")


def score_to_mastery_tier(score: float) -> str:
    """Maps a numeric mastery score in [0.0, 1.0] to a canonical tier label."""
    if score >= 0.80:
        return "Hero"
    if score >= 0.45:
        return "Practitioner"
    if score > 0.0:
        return "Novice"
    return "Unexplored"


@dataclass
class ConceptNode:
    """Represents a single AI concept vertex in the learner's mastery DAG."""

    concept_id: str
    title: str
    category: str
    mastery_score: float = 0.25
    mastery_tier: str = "Novice"
    prerequisites: List[str] = field(default_factory=list)
    key_equation: str = ""
    embedding_3d: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    interactions_count: int = 1
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        self.mastery_score = round(max(0.0, min(1.0, float(self.mastery_score))), 3)
        self.mastery_tier = score_to_mastery_tier(self.mastery_score)
        if not self.embedding_3d or self.embedding_3d == [0.0, 0.0, 0.0]:
            self.embedding_3d = compute_concept_embedding(self.concept_id, dims=3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "title": self.title,
            "category": self.category,
            "mastery_score": self.mastery_score,
            "mastery_tier": self.mastery_tier,
            "prerequisites": list(self.prerequisites),
            "key_equation": self.key_equation,
            "embedding_3d": list(self.embedding_3d),
            "interactions_count": self.interactions_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConceptNode:
        return cls(
            concept_id=str(data["concept_id"]),
            title=str(data.get("title", data["concept_id"])),
            category=str(data.get("category", "Deep Learning")),
            mastery_score=float(data.get("mastery_score", 0.25)),
            mastery_tier=str(data.get("mastery_tier", "Novice")),
            prerequisites=list(data.get("prerequisites", [])),
            key_equation=str(data.get("key_equation", "")),
            embedding_3d=list(data.get("embedding_3d", [0.0, 0.0, 0.0])),
            interactions_count=int(data.get("interactions_count", 1)),
            updated_at=str(
                data.get("updated_at", datetime.now(timezone.utc).isoformat())
            ),
        )


@dataclass
class ConceptEdge:
    """Directed edge from a prerequisite concept to a dependent concept in the DAG."""

    source_id: str
    target_id: str
    relationship: str = "prerequisite_for"
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship,
            "weight": self.weight,
        }


DEFAULT_FOUNDATIONAL_NODES: Tuple[Dict[str, Any], ...] = (
    {
        "concept_id": "linear_algebra_cosine_similarity",
        "title": "Cosine Similarity & Vector Spaces",
        "category": "Foundations",
        "mastery_score": 0.92,
        "prerequisites": [],
        "key_equation": "s(u, v) = (u · v) / (||u||_2 ||v||_2)",
        "embedding_3d": [-0.62, 0.48, 0.62],
    },
    {
        "concept_id": "self_attention_mechanism",
        "title": "Scaled Dot-Product Attention",
        "category": "Sequence Modeling",
        "mastery_score": 0.88,
        "prerequisites": ["linear_algebra_cosine_similarity"],
        "key_equation": "Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V",
        "embedding_3d": [-0.24, 0.71, 0.66],
    },
    {
        "concept_id": "transformer_architecture",
        "title": "Transformer Architecture",
        "category": "Foundation Models",
        "mastery_score": 0.85,
        "prerequisites": ["self_attention_mechanism"],
        "key_equation": "h_l = LayerNorm(h_{l-1} + MHA(h_{l-1}))",
        "embedding_3d": [0.18, 0.78, 0.60],
    },
    {
        "concept_id": "vector_quantization_vqvae",
        "title": "Discrete Tokenization (VQ-VAE)",
        "category": "Generative Modeling",
        "mastery_score": 0.64,
        "prerequisites": ["linear_algebra_cosine_similarity"],
        "key_equation": "z_q(x) = argmin_{e_k in E} ||z_e(x) - e_k||_2",
        "embedding_3d": [0.51, 0.44, 0.74],
    },
)


class LearnerKnowledgeGraph:
    """Manages learner Directed Acyclic Graphs (DAGs) with cycle prevention."""

    def __init__(self, store: Optional[FirestoreSessionStore] = None) -> None:
        self.store = store or get_firestore_store()

    def _seed_default_dag(self, learner_id: str) -> Dict[str, ConceptNode]:
        nodes: Dict[str, ConceptNode] = {}
        for item in DEFAULT_FOUNDATIONAL_NODES:
            node = ConceptNode.from_dict(item)
            nodes[node.concept_id] = node
        return nodes

    def load_nodes(self, learner_id: str) -> Dict[str, ConceptNode]:
        """Loads a learner's DAG nodes from Firestore or initializes defaults."""
        persisted = self.store.get_learner_dag(learner_id)
        if not persisted or "nodes" not in persisted:
            nodes = self._seed_default_dag(learner_id)
            self.store.save_learner_dag(
                learner_id,
                self._serialize_dag(learner_id, nodes),
            )
            return nodes

        raw_nodes = persisted.get("nodes", {})
        nodes: Dict[str, ConceptNode] = {}
        for cid, payload in raw_nodes.items():
            nodes[cid] = ConceptNode.from_dict(payload)
        return nodes

    @staticmethod
    def _would_create_cycle(
        nodes: Dict[str, ConceptNode],
        new_node_id: str,
        prerequisites: List[str],
    ) -> bool:
        """Verifies that adding edges `prereq -> new_node_id` preserves DAG acyclicity."""
        if new_node_id in prerequisites:
            return True

        # Build adjacency list: prereq -> list of dependents
        adj: Dict[str, Set[str]] = {cid: set() for cid in nodes}
        adj.setdefault(new_node_id, set())
        for cid, node in nodes.items():
            for prereq in node.prerequisites:
                adj.setdefault(prereq, set()).add(cid)
        for prereq in prerequisites:
            adj.setdefault(prereq, set()).add(new_node_id)

        # Kahn's / DFS cycle check
        visited: Dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=visited

        def dfs(curr: str) -> bool:
            state = visited.get(curr, 0)
            if state == 1:
                return True
            if state == 2:
                return False
            visited[curr] = 1
            for nxt in adj.get(curr, set()):
                if dfs(nxt):
                    return True
            visited[curr] = 2
            return False

        for vertex in list(adj.keys()):
            if visited.get(vertex, 0) == 0:
                if dfs(vertex):
                    return True
        return False

    def topological_order(self, nodes: Dict[str, ConceptNode]) -> List[str]:
        """Returns concept_ids in valid topological order (prerequisites first)."""
        in_degree: Dict[str, int] = {cid: 0 for cid in nodes}
        dependents: Dict[str, List[str]] = {cid: [] for cid in nodes}

        for cid, node in nodes.items():
            for prereq in node.prerequisites:
                if prereq in nodes:
                    in_degree[cid] += 1
                    dependents[prereq].append(cid)

        queue: List[str] = sorted([cid for cid, deg in in_degree.items() if deg == 0])
        ordered: List[str] = []

        while queue:
            curr = queue.pop(0)
            ordered.append(curr)
            for dep in sorted(dependents.get(curr, [])):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        return ordered

    def _serialize_dag(
        self, learner_id: str, nodes: Dict[str, ConceptNode]
    ) -> Dict[str, Any]:
        edges: List[Dict[str, Any]] = []
        for cid, node in nodes.items():
            for prereq in node.prerequisites:
                if prereq in nodes:
                    edges.append(
                        ConceptEdge(
                            source_id=prereq,
                            target_id=cid,
                            relationship="prerequisite_for",
                        ).to_dict()
                    )

        topo = self.topological_order(nodes)
        avg_mastery = (
            round(
                sum(n.mastery_score for n in nodes.values()) / max(1, len(nodes)),
                3,
            )
            if nodes
            else 0.0
        )
        return {
            "learner_id": learner_id,
            "is_valid_dag": len(topo) == len(nodes),
            "topological_order": topo,
            "average_mastery": avg_mastery,
            "nodes": {cid: n.to_dict() for cid, n in nodes.items()},
            "edges": edges,
        }

    def query_graph(
        self,
        learner_id: str,
        target_concept_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Queries the learner's DAG and recommends the optimal pedagogical tier."""
        nodes = self.load_nodes(learner_id)
        serialized = self._serialize_dag(learner_id, nodes)

        recommended_tier = "Novice"
        prereq_readiness = 0.85

        if target_concept_id and target_concept_id in nodes:
            target_node = nodes[target_concept_id]
            recommended_tier = target_node.mastery_tier
            if target_node.prerequisites:
                scores = [
                    nodes[p].mastery_score
                    for p in target_node.prerequisites
                    if p in nodes
                ]
                if scores:
                    prereq_readiness = round(sum(scores) / len(scores), 3)
        elif nodes:
            # Infer readiness from Transformer Architecture & Self-Attention mastery
            tf_node = nodes.get("transformer_architecture")
            if tf_node and tf_node.mastery_score >= 0.80:
                prereq_readiness = tf_node.mastery_score

        serialized["recommended_starting_tier"] = recommended_tier
        serialized["prerequisite_readiness_score"] = prereq_readiness
        return serialized

    def upsert_concept_mastery(
        self,
        *,
        learner_id: str,
        concept_id: str,
        title: str,
        category: str = "Frontier AI",
        mastery_delta: float = 0.25,
        absolute_mastery: Optional[float] = None,
        prerequisites: Optional[List[str]] = None,
        key_equation: str = "",
    ) -> Dict[str, Any]:
        """Updates or inserts a concept node in the learner's DAG asynchronously.

        Validates strict DAG acyclicity before mutating edges and dispatches
        persistence + vector indexing via `schedule_async_dag_update`.

        Raises:
            ValueError: If the requested prerequisite edges would introduce a cycle.
        """
        nodes = self.load_nodes(learner_id)
        existing = nodes.get(concept_id)

        candidate_prereqs = (
            list(prerequisites)
            if prerequisites is not None
            else (existing.prerequisites if existing else ["transformer_architecture"])
        )
        # Filter to known or valid node identifiers
        valid_prereqs = [p for p in candidate_prereqs if p in nodes and p != concept_id]
        if prerequisites is not None and concept_id in prerequisites:
            raise ValueError(
                f"Self-dependency cycle detected: '{concept_id}' cannot be its own prerequisite."
            )

        if self._would_create_cycle(nodes, concept_id, candidate_prereqs):
            raise ValueError(
                f"Cycle detected in Learner Knowledge Graph DAG when linking {candidate_prereqs} -> {concept_id}."
            )

        if existing is not None:
            new_score = (
                absolute_mastery
                if absolute_mastery is not None
                else min(1.0, max(0.0, existing.mastery_score + mastery_delta))
            )
            updated_node = ConceptNode(
                concept_id=concept_id,
                title=title or existing.title,
                category=category or existing.category,
                mastery_score=new_score,
                prerequisites=valid_prereqs or existing.prerequisites,
                key_equation=key_equation or existing.key_equation,
                interactions_count=existing.interactions_count + 1,
            )
        else:
            initial_score = (
                absolute_mastery
                if absolute_mastery is not None
                else min(1.0, max(0.15, 0.35 + mastery_delta))
            )
            updated_node = ConceptNode(
                concept_id=concept_id,
                title=title,
                category=category,
                mastery_score=initial_score,
                prerequisites=valid_prereqs or ["transformer_architecture"],
                key_equation=key_equation,
                interactions_count=1,
            )

        nodes[concept_id] = updated_node
        dag_snapshot = self._serialize_dag(learner_id, nodes)

        # Schedule non-blocking async Firestore write & embedding update
        self.store.schedule_async_dag_update(
            learner_id=learner_id,
            concept_id=concept_id,
            mastery_score=updated_node.mastery_score,
            dag_snapshot=dag_snapshot,
        )

        return {
            "updated_node": updated_node.to_dict(),
            "dag": dag_snapshot,
            "async_persistence_scheduled": True,
        }


_GLOBAL_GRAPH = LearnerKnowledgeGraph()


def get_knowledge_graph() -> LearnerKnowledgeGraph:
    """Returns the singleton LearnerKnowledgeGraph instance."""
    return _GLOBAL_GRAPH
