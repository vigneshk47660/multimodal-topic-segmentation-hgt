"""
Stage 1: Lecture Transcript Ingestion using Instructional Unit Builder
Implements Eq. 1-2 from the paper.
Converts raw lecture transcript L into ordered instructional units U = {u1, u2, ..., uN}
Each unit u_i = <c_i, t_i> where c_i is content and t_i is temporal index.
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
from enum import Enum


class ModalityType(Enum):
    TEXT = "text"
    EQUATION = "equation"
    TABLE = "table"
    DIAGRAM = "diagram"
    UNKNOWN = "unknown"


@dataclass
class InstructionalUnit:
    """Represents a minimal instructional element u_i = <c_i, t_i> (Eq. 2)"""
    content: str
    temporal_index: int
    modality: ModalityType = ModalityType.UNKNOWN
    normalized_content: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["modality"] = self.modality.value
        return d


class InstructionalUnitBuilder:
    """
    Rule-based segmentation that breaks a continuous lecture transcript
    into minimal instructional units while maintaining temporal order.
    L -> U = {u1, u2, ..., uN}  (Eq. 1)
    """

    EQUATION_PATTERNS = [
        r'\$\$.+?\$\$',
        r'\$[^$]+?\$',
        r'\\begin\{equation\}.*?\\end\{equation\}',
        r'\\begin\{align\}.*?\\end\{align\}',
        r'\\begin\{math\}.*?\\end\{math\}',
        r'\\[.*?\\]',
        r'\\(.*?\\)',
        r'(?:^|\s)[A-Za-z]\s*=\s*[^,\.\n]+',
        r'(?:^|\s)(?:∑|∏|∫|∂|∇|Δ|λ|α|β|γ|θ|σ|μ|π)[^\s]*\s*[=<>≤≥]+\s*[^\s,\.]+',
        r'f\s*\([^)]+\)\s*=\s*[^\n]+',
    ]

    TABLE_PATTERNS = [
        r'\\begin\{tabular\}.*?\\end\{tabular\}',
        r'\\begin\{table\}.*?\\end\{table\}',
        r'\|(?:[^|\n]+\|){2,}',
        r'(?:Table|TABLE)\s*\d+[\.:]\s*[^\n]+',
        r'(?:\+[-=]+){2,}\+',
    ]

    DIAGRAM_PATTERNS = [
        r'(?:Fig(?:ure)?|FIGURE)\s*\.?\s*\d+[\.:]\s*[^\n]+',
        r'\\begin\{figure\}.*?\\end\{figure\}',
        r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}',
        r'\[(?:IMAGE|DIAGRAM|FIGURE|GRAPH|CHART|PLOT)(?:\s*:\s*[^\]]+)?\]',
        r'(?:see|refer(?:\s+to)?|as\s+(?:shown|illustrated)\s+in)\s+(?:Fig(?:ure)?|Diagram)\s*\.?\s*\d+',
    ]

    def __init__(self):
        self._eq_patterns = [re.compile(p, re.DOTALL | re.MULTILINE) for p in self.EQUATION_PATTERNS]
        self._table_patterns = [re.compile(p, re.DOTALL | re.MULTILINE) for p in self.TABLE_PATTERNS]
        self._diagram_patterns = [re.compile(p, re.DOTALL | re.MULTILINE) for p in self.DIAGRAM_PATTERNS]

    def _detect_modality(self, text: str) -> ModalityType:
        text_stripped = text.strip()
        for pattern in self._eq_patterns:
            if pattern.search(text_stripped):
                return ModalityType.EQUATION
        for pattern in self._table_patterns:
            if pattern.search(text_stripped):
                return ModalityType.TABLE
        for pattern in self._diagram_patterns:
            if pattern.search(text_stripped):
                return ModalityType.DIAGRAM
        return ModalityType.TEXT

    def _split_by_markers(self, transcript: str) -> List[Tuple[str, ModalityType]]:
        """Split transcript into segments based on structural and temporal markers."""
        segments = []
        all_patterns = []

        for p in self._eq_patterns:
            for m in p.finditer(transcript):
                all_patterns.append((m.start(), m.end(), ModalityType.EQUATION, m.group()))
        for p in self._table_patterns:
            for m in p.finditer(transcript):
                all_patterns.append((m.start(), m.end(), ModalityType.TABLE, m.group()))
        for p in self._diagram_patterns:
            for m in p.finditer(transcript):
                all_patterns.append((m.start(), m.end(), ModalityType.DIAGRAM, m.group()))

        all_patterns.sort(key=lambda x: x[0])

        # Remove overlapping matches
        filtered = []
        last_end = -1
        for start, end, mod, text in all_patterns:
            if start >= last_end:
                filtered.append((start, end, mod, text))
                last_end = end

        # Build segments: text between special elements + the special elements
        prev_end = 0
        for start, end, mod, matched_text in filtered:
            if start > prev_end:
                text_between = transcript[prev_end:start].strip()
                if text_between:
                    for sent in self._split_text_into_sentences(text_between):
                        if sent.strip():
                            segments.append((sent.strip(), ModalityType.TEXT))
            segments.append((matched_text.strip(), mod))
            prev_end = end

        # Remaining text after last special element
        if prev_end < len(transcript):
            remaining = transcript[prev_end:].strip()
            if remaining:
                for sent in self._split_text_into_sentences(remaining):
                    if sent.strip():
                        segments.append((sent.strip(), ModalityType.TEXT))

        return segments

    def _split_text_into_sentences(self, text: str) -> List[str]:
        """Split text into sentence-level instructional units."""
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        result = []
        for s in sentences:
            s = s.strip()
            if len(s) > 500:
                sub = re.split(r'[;:]\s+', s)
                result.extend([x.strip() for x in sub if x.strip()])
            elif s:
                result.append(s)
        return result

    def build(self, transcript: str) -> List[InstructionalUnit]:
        """
        Convert lecture transcript L into ordered instructional units.
        L -> U = {u1, u2, ..., uN}  (Eq. 1)
        Each u_i = <c_i, t_i>        (Eq. 2)
        """
        segments = self._split_by_markers(transcript)
        units = []
        for idx, (content, modality) in enumerate(segments):
            unit = InstructionalUnit(
                content=content,
                temporal_index=idx,
                modality=modality,
                metadata={"raw_length": len(content)}
            )
            units.append(unit)
        return units

    def build_from_structured(self, structured_data: List[dict]) -> List[InstructionalUnit]:
        """
        Build units from pre-structured data (e.g., JSON with content and modality fields).
        """
        units = []
        for idx, item in enumerate(structured_data):
            content = item.get("content", "")
            mod_str = item.get("modality", "text").lower()
            try:
                modality = ModalityType(mod_str)
            except ValueError:
                modality = self._detect_modality(content)

            unit = InstructionalUnit(
                content=content,
                temporal_index=idx,
                modality=modality,
                metadata=item.get("metadata", {})
            )
            units.append(unit)
        return units

    def build_from_file(self, filepath: str) -> List[InstructionalUnit]:
        """Load transcript from file and build instructional units."""
        if filepath.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return self.build_from_structured(data)
            elif isinstance(data, dict) and "transcript" in data:
                return self.build(data["transcript"])
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                transcript = f.read()
            return self.build(transcript)


def build_instructional_units(transcript: str) -> List[InstructionalUnit]:
    """Convenience function for Stage 1."""
    builder = InstructionalUnitBuilder()
    return builder.build(transcript)
