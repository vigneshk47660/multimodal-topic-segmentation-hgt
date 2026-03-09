"""
Stage 2: Preprocessing and Modality Identification using HLC Normalization
Implements Eq. 3 from the paper.
Transforms each unit into u_tilde_i = <c_tilde_i, t_i, m_i>
where c_tilde_i is normalized content, t_i is temporal index, m_i is modality label.
"""

import re
import unicodedata
from typing import List, Optional
from models.instructional_unit_builder import InstructionalUnit, ModalityType


class HLCNormalizer:
    """
    Modality-aware normalization that prepares instructional units
    for representation learning while preserving semantic and structural characteristics.
    """

    MATH_SYMBOL_MAP = {
        '×': '*', '÷': '/', '−': '-', '±': '+-',
        '≤': '<=', '≥': '>=', '≠': '!=', '≈': '~=',
        '∞': 'inf', '√': 'sqrt', 'π': 'pi',
        '∑': 'sum', '∏': 'prod', '∫': 'integral',
        '∂': 'partial', '∇': 'nabla', 'Δ': 'Delta',
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma',
        'δ': 'delta', 'ε': 'epsilon', 'ζ': 'zeta',
        'η': 'eta', 'θ': 'theta', 'λ': 'lambda',
        'μ': 'mu', 'σ': 'sigma', 'τ': 'tau',
        'φ': 'phi', 'ψ': 'psi', 'ω': 'omega',
    }

    def __init__(self, max_text_length: int = 512, max_eq_length: int = 256,
                 max_table_cells: int = 100):
        self.max_text_length = max_text_length
        self.max_eq_length = max_eq_length
        self.max_table_cells = max_table_cells

    def _normalize_text(self, content: str) -> str:
        text = content.strip()
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,;:!?\'\"()\-/]', '', text)
        text = text.lower()
        words = text.split()
        if len(words) > self.max_text_length:
            words = words[:self.max_text_length]
        return ' '.join(words)

    def _normalize_equation(self, content: str) -> str:
        eq = content.strip()
        for delim in [('$$', '$$'), ('$', '$'), ('\\[', '\\]'), ('\\(', '\\)')]:
            if eq.startswith(delim[0]) and eq.endswith(delim[1]):
                eq = eq[len(delim[0]):-len(delim[1])].strip()
                break
        for env in ['equation', 'align', 'math', 'gather', 'multline']:
            eq = re.sub(rf'\\begin\{{{env}\*?\}}', '', eq)
            eq = re.sub(rf'\\end\{{{env}\*?\}}', '', eq)
        eq = re.sub(r'\\label\{[^}]*\}', '', eq)
        eq = re.sub(r'\\tag\{[^}]*\}', '', eq)
        for sym, replacement in self.MATH_SYMBOL_MAP.items():
            eq = eq.replace(sym, f' {replacement} ')
        eq = re.sub(r'\s+', ' ', eq).strip()
        if len(eq) > self.max_eq_length:
            eq = eq[:self.max_eq_length]
        return eq

    def _normalize_table(self, content: str) -> str:
        table = content.strip()
        for env in ['table', 'tabular']:
            table = re.sub(rf'\\begin\{{{env}\*?\}}(?:\[[^\]]*\])?(?:\{{[^}}]*\}})?', '', table)
            table = re.sub(rf'\\end\{{{env}\*?\}}', '', table)
        table = re.sub(r'\\(?:hline|toprule|midrule|bottomrule|cline\{[^}]*\})', '', table)
        table = re.sub(r'\\caption\{([^}]*)\}', r'Caption: \1', table)
        rows = re.split(r'\\\\|\n', table)
        cells = []
        for row in rows:
            row_cells = re.split(r'&|\|', row)
            row_cells = [c.strip() for c in row_cells if c.strip()]
            if row_cells:
                cells.extend(row_cells)
        if len(cells) > self.max_table_cells:
            cells = cells[:self.max_table_cells]
        return ' [SEP] '.join(cells) if cells else table

    def _normalize_diagram(self, content: str) -> str:
        diag = content.strip()
        caption_match = re.search(
            r'(?:Fig(?:ure)?|FIGURE)\s*\.?\s*(\d+)[\.:]\s*(.+?)(?:\.|$)',
            diag, re.IGNORECASE
        )
        if caption_match:
            fig_num = caption_match.group(1)
            caption = caption_match.group(2).strip()
            return f"Figure {fig_num}: {caption}"
        ref_match = re.search(
            r'\[(?:IMAGE|DIAGRAM|FIGURE|GRAPH|CHART|PLOT)(?:\s*:\s*([^\]]+))?\]',
            diag, re.IGNORECASE
        )
        if ref_match:
            desc = ref_match.group(1) or "visual content"
            return f"Diagram: {desc.strip()}"
        return f"Visual: {diag[:200]}"

    def _refine_modality(self, unit: InstructionalUnit) -> ModalityType:
        """Verify and potentially correct modality assignment."""
        content = unit.content.strip()
        if unit.modality != ModalityType.UNKNOWN:
            return unit.modality

        eq_indicators = ['=', '\\frac', '\\sum', '\\int', '$', '\\begin{equation}']
        if any(ind in content for ind in eq_indicators):
            return ModalityType.EQUATION

        table_indicators = ['|', '\\begin{tabular}', '&', '\\hline']
        table_count = sum(1 for ind in table_indicators if ind in content)
        if table_count >= 2:
            return ModalityType.TABLE

        if re.search(r'(?:Fig|Figure|Diagram|Graph|Chart|Plot)\s*\.?\s*\d+', content, re.IGNORECASE):
            return ModalityType.DIAGRAM

        return ModalityType.TEXT

    def normalize(self, units: List[InstructionalUnit]) -> List[InstructionalUnit]:
        """
        Normalize and label all instructional units.
        u_tilde_i = <c_tilde_i, t_i, m_i>  (Eq. 3)
        """
        normalized_units = []
        for unit in units:
            modality = self._refine_modality(unit)
            if modality == ModalityType.TEXT:
                norm_content = self._normalize_text(unit.content)
            elif modality == ModalityType.EQUATION:
                norm_content = self._normalize_equation(unit.content)
            elif modality == ModalityType.TABLE:
                norm_content = self._normalize_table(unit.content)
            elif modality == ModalityType.DIAGRAM:
                norm_content = self._normalize_diagram(unit.content)
            else:
                norm_content = self._normalize_text(unit.content)
                modality = ModalityType.TEXT

            norm_unit = InstructionalUnit(
                content=unit.content,
                temporal_index=unit.temporal_index,
                modality=modality,
                normalized_content=norm_content,
                metadata={**unit.metadata, "original_modality": unit.modality.value}
            )
            normalized_units.append(norm_unit)
        return normalized_units


def normalize_and_label(units: List[InstructionalUnit],
                        max_text_length: int = 512) -> List[InstructionalUnit]:
    """Convenience function for Stage 2."""
    normalizer = HLCNormalizer(max_text_length=max_text_length)
    return normalizer.normalize(units)
