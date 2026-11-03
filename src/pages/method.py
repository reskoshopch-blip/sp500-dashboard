from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.metrics import METRICS
from src.config import PROJECT_ROOT
from src.normalization.taxonomy import ITEMS, overlap_documentation


def page() -> None:
    st.title("Methodik und Quellen")
    doc = PROJECT_ROOT / "docs" / "METHODIK.md"
    if doc.exists():
        st.markdown(doc.read_text(encoding="utf-8"))
    st.markdown("### Positionen und XBRL-Tags")
    st.dataframe(pd.DataFrame([{"Abschnitt": i.section, "Position": i.label_de, "Englisch": i.label_en,
                                "Ebene": i.level, "XBRL-Tags (Priorität)": " → ".join(i.tags), "Hinweis": i.note}
                               for i in ITEMS]), hide_index=True, width="stretch", height=420)
    st.markdown("### Überlappende Positionen (nie gemeinsam addieren)")
    st.dataframe(pd.DataFrame(overlap_documentation()), hide_index=True, width="stretch")
    st.markdown("### Kennzahlen")
    st.dataframe(pd.DataFrame([{"Kennzahl": m.label, "Schlüssel": m.key, "Formel / Regel": m.formula}
                               for m in METRICS]), hide_index=True, width="stretch")
