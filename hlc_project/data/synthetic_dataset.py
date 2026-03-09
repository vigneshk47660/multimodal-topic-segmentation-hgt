"""
Synthetic Heterogeneous Lecture Content (HLC) Dataset Generator
Generates lecture transcripts with text, equations, tables, and diagrams
with known ground-truth topic boundaries for evaluation.
"""

import json
import os
import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

from models.instructional_unit_builder import ModalityType


TOPIC_POOLS = {
    "computer_science": {
        "topics": [
            "Introduction to Algorithms",
            "Data Structures Overview",
            "Sorting Algorithms",
            "Graph Theory",
            "Dynamic Programming",
            "Machine Learning Basics",
            "Neural Networks",
            "Optimization Methods",
            "Complexity Theory",
            "Database Systems",
        ],
        "text_templates": [
            "In this section, we will discuss {topic}. This is a fundamental concept in computer science.",
            "The key idea behind {topic} is to decompose problems into smaller sub-problems.",
            "Let us now examine the theoretical foundations of {topic}.",
            "{topic} has numerous applications in modern computing systems.",
            "Understanding {topic} requires knowledge of mathematical reasoning and logical thinking.",
            "The historical development of {topic} can be traced back to early computational theory.",
            "Practical implementations of {topic} often involve trade-offs between time and space complexity.",
            "Recent advances in {topic} have led to significant breakthroughs in performance.",
        ],
        "equation_templates": [
            "T(n) = 2T(n/2) + O(n)",
            "f(x) = sum_{i=1}^{n} w_i * x_i + b",
            "L(theta) = -sum_{i} y_i log(p_i) + (1-y_i) log(1-p_i)",
            "O(n log n) = O(n) * O(log n)",
            "P(A|B) = P(B|A) * P(A) / P(B)",
            "gradient = partial L / partial w",
            "h(x) = sigma(W^T x + b)",
            "E[X] = sum_{i} x_i * P(x_i)",
        ],
        "table_templates": [
            "Algorithm | Time Complexity | Space Complexity\nBubble Sort | O(n^2) | O(1)\nMerge Sort | O(n log n) | O(n)\nQuick Sort | O(n log n) | O(log n)",
            "Operation | Array | Linked List | Hash Table\nSearch | O(n) | O(n) | O(1)\nInsert | O(n) | O(1) | O(1)\nDelete | O(n) | O(1) | O(1)",
            "Dataset | Accuracy | Precision | Recall | F1\nMNIST | 99.2 | 99.1 | 99.3 | 99.2\nCIFAR-10 | 93.5 | 93.2 | 93.8 | 93.5",
        ],
        "diagram_templates": [
            "Figure {n}: Architecture of the neural network with input, hidden, and output layers.",
            "Figure {n}: Comparison of sorting algorithms performance on datasets of varying sizes.",
            "Figure {n}: Graph representation showing nodes and weighted edges.",
            "Figure {n}: Decision tree structure for classification task.",
            "Figure {n}: Flowchart of the proposed algorithm pipeline.",
        ],
    },
    "mathematics": {
        "topics": [
            "Linear Algebra Fundamentals",
            "Calculus and Integration",
            "Probability Distributions",
            "Matrix Operations",
            "Differential Equations",
            "Fourier Analysis",
            "Optimization Theory",
            "Number Theory",
        ],
        "text_templates": [
            "The concept of {topic} is essential for understanding advanced mathematics.",
            "We begin our study of {topic} with the basic definitions and axioms.",
            "Applications of {topic} span engineering, physics, and economics.",
            "{topic} provides the mathematical framework for modeling real-world phenomena.",
            "The proof of the main theorem in {topic} follows from these lemmas.",
        ],
        "equation_templates": [
            "integral_a^b f(x) dx = F(b) - F(a)",
            "det(A) = sum_{sigma} sgn(sigma) prod_{i} a_{i,sigma(i)}",
            "nabla f(x) = (partial f/partial x_1, ..., partial f/partial x_n)",
            "A^{-1} = (1/det(A)) * adj(A)",
            "E[X^2] - (E[X])^2 = Var(X)",
            "f(x) = sum_{n=0}^{infinity} a_n * (x - c)^n",
        ],
        "table_templates": [
            "Distribution | Mean | Variance | PDF\nNormal | mu | sigma^2 | (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))\nPoisson | lambda | lambda | (lambda^k * e^{-lambda})/k!",
            "Matrix Property | Definition | Example\nSymmetric | A = A^T | [[1,2],[2,3]]\nOrthogonal | A^T*A = I | Rotation matrices",
        ],
        "diagram_templates": [
            "Figure {n}: Visualization of the probability density function for various distributions.",
            "Figure {n}: Geometric interpretation of eigenvalues and eigenvectors.",
            "Figure {n}: Surface plot of the objective function with gradient descent path.",
        ],
    },
    "physics": {
        "topics": [
            "Classical Mechanics",
            "Electromagnetism",
            "Thermodynamics",
            "Quantum Mechanics",
            "Wave Theory",
            "Relativity",
        ],
        "text_templates": [
            "The study of {topic} begins with fundamental physical laws and principles.",
            "In {topic}, we observe that energy conservation plays a critical role.",
            "{topic} describes the behavior of matter and energy at various scales.",
            "Experimental verification of {topic} has been confirmed through numerous studies.",
        ],
        "equation_templates": [
            "F = m * a",
            "E = m * c^2",
            "psi(x,t) = A * exp(i*(k*x - omega*t))",
            "nabla x E = -partial B/partial t",
            "dS >= delta Q / T",
        ],
        "table_templates": [
            "Particle | Mass (MeV) | Charge | Spin\nElectron | 0.511 | -1 | 1/2\nProton | 938.3 | +1 | 1/2\nNeutron | 939.6 | 0 | 1/2",
        ],
        "diagram_templates": [
            "Figure {n}: Free body diagram showing forces acting on the object.",
            "Figure {n}: Wave interference pattern in double-slit experiment.",
            "Figure {n}: Phase diagram of water showing solid, liquid, and gas regions.",
        ],
    },
}


@dataclass
class SyntheticLecture:
    lecture_id: str
    domain: str
    units: List[Dict]
    ground_truth_boundaries: List[int]
    topic_labels: List[str]
    num_units: int
    num_topics: int

    def to_dict(self):
        return asdict(self)


class SyntheticHLCDatasetGenerator:
    """Generate synthetic heterogeneous lecture content with ground-truth boundaries."""

    def __init__(
        self,
        num_lectures: int = 100,
        min_units: int = 30,
        max_units: int = 80,
        min_topics: int = 3,
        max_topics: int = 8,
        modality_dist: Optional[Dict[str, float]] = None,
        noise_ratio: float = 0.05,
        seed: int = 42,
    ):
        self.num_lectures = num_lectures
        self.min_units = min_units
        self.max_units = max_units
        self.min_topics = min_topics
        self.max_topics = max_topics
        self.modality_dist = modality_dist or {
            "text": 0.50, "equation": 0.20, "table": 0.15, "diagram": 0.15
        }
        self.noise_ratio = noise_ratio
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.fig_counter = 0

    def _sample_modality(self) -> str:
        mods = list(self.modality_dist.keys())
        probs = [self.modality_dist[m] for m in mods]
        return self.rng.choices(mods, weights=probs, k=1)[0]

    def _generate_unit(self, domain: str, topic: str, modality: str) -> Dict:
        pool = TOPIC_POOLS[domain]
        if modality == "text":
            template = self.rng.choice(pool["text_templates"])
            content = template.format(topic=topic)
        elif modality == "equation":
            content = self.rng.choice(pool["equation_templates"])
        elif modality == "table":
            content = self.rng.choice(pool["table_templates"])
        elif modality == "diagram":
            self.fig_counter += 1
            template = self.rng.choice(pool["diagram_templates"])
            content = template.format(n=self.fig_counter)
        else:
            content = f"Content about {topic}"

        return {"content": content, "modality": modality, "topic": topic}

    def generate_lecture(self, lecture_id: str) -> SyntheticLecture:
        domain = self.rng.choice(list(TOPIC_POOLS.keys()))
        pool = TOPIC_POOLS[domain]

        num_topics = self.rng.randint(self.min_topics, self.max_topics)
        topics = self.rng.sample(pool["topics"], min(num_topics, len(pool["topics"])))

        total_units = self.rng.randint(self.min_units, self.max_units)
        units_per_topic = self._distribute_units(total_units, len(topics))

        all_units = []
        boundaries = []
        topic_labels = []
        current_idx = 0

        for topic_idx, (topic, n_units) in enumerate(zip(topics, units_per_topic)):
            if topic_idx > 0:
                boundaries.append(current_idx)

            # Generate units for this topic
            # First unit of each topic is always text (topic introduction)
            first_unit = self._generate_unit(domain, topic, "text")
            first_unit["temporal_index"] = current_idx
            all_units.append(first_unit)
            current_idx += 1

            for _ in range(n_units - 1):
                modality = self._sample_modality()
                unit = self._generate_unit(domain, topic, modality)
                unit["temporal_index"] = current_idx

                # Add noise: occasionally assign wrong topic content
                if self.rng.random() < self.noise_ratio:
                    noise_topic = self.rng.choice(pool["topics"])
                    unit["content"] = unit["content"].replace(topic, noise_topic)
                    unit["metadata"] = {"noisy": True}

                all_units.append(unit)
                current_idx += 1

            topic_labels.append(topic)

        return SyntheticLecture(
            lecture_id=lecture_id,
            domain=domain,
            units=all_units,
            ground_truth_boundaries=boundaries,
            topic_labels=topic_labels,
            num_units=len(all_units),
            num_topics=len(topics),
        )

    def _distribute_units(self, total: int, n_topics: int) -> List[int]:
        """Distribute units across topics with some variance."""
        base = total // n_topics
        remainder = total % n_topics
        distribution = [base] * n_topics
        for i in range(remainder):
            distribution[i] += 1
        # Add variance
        for i in range(n_topics):
            delta = self.rng.randint(-2, 2)
            distribution[i] = max(3, distribution[i] + delta)
        return distribution

    def generate_dataset(self, output_dir: str) -> Dict:
        """Generate full synthetic dataset and save to disk."""
        os.makedirs(output_dir, exist_ok=True)

        lectures = []
        all_stats = {
            "total_lectures": self.num_lectures,
            "total_units": 0,
            "total_boundaries": 0,
            "modality_counts": {"text": 0, "equation": 0, "table": 0, "diagram": 0},
            "domain_counts": {},
        }

        for i in range(self.num_lectures):
            lecture = self.generate_lecture(f"lecture_{i:04d}")
            lectures.append(lecture.to_dict())

            all_stats["total_units"] += lecture.num_units
            all_stats["total_boundaries"] += len(lecture.ground_truth_boundaries)
            all_stats["domain_counts"][lecture.domain] = (
                all_stats["domain_counts"].get(lecture.domain, 0) + 1
            )
            for unit in lecture.units:
                mod = unit["modality"]
                all_stats["modality_counts"][mod] = (
                    all_stats["modality_counts"].get(mod, 0) + 1
                )

        # Split into train/val/test (70/15/15)
        self.rng.shuffle(lectures)
        n = len(lectures)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        splits = {
            "train": lectures[:train_end],
            "val": lectures[train_end:val_end],
            "test": lectures[val_end:],
        }

        for split_name, split_data in splits.items():
            filepath = os.path.join(output_dir, f"{split_name}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)

        # Save statistics
        all_stats["splits"] = {k: len(v) for k, v in splits.items()}
        stats_path = os.path.join(output_dir, "dataset_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, indent=2)

        return all_stats


def generate_synthetic_dataset(
    output_dir: str = "./data/synthetic_hlc",
    num_lectures: int = 100,
    seed: int = 42,
) -> Dict:
    """Convenience function to generate synthetic HLC dataset."""
    generator = SyntheticHLCDatasetGenerator(
        num_lectures=num_lectures, seed=seed
    )
    return generator.generate_dataset(output_dir)


if __name__ == "__main__":
    stats = generate_synthetic_dataset("./data/synthetic_hlc", num_lectures=100)
    print(json.dumps(stats, indent=2))
