"""Kleine Hilfsfunktionen."""
from __future__ import annotations

import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)


def atomic_write_text(path: Path, text: str, attempts: int = 20) -> bool:
    """Schreibt erst eine temporäre Datei und ersetzt dann das Ziel.
    Unter Windows kann das Ziel kurz gesperrt sein, wenn das Dashboard es gerade liest (WinError 5/32) –
    dann wird mehrmals neu versucht. Gelingt es nicht, wird nur protokolliert: Ein Statusfile darf nie den
    eigentlichen Datenabruf abbrechen."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    for i in range(attempts):
        try:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
            return True
        except PermissionError:
            time.sleep(0.1 * (i + 1))
        except OSError as exc:
            log.warning("Konnte %s nicht schreiben: %s", path, exc)
            return False
    log.warning("Datei %s war gesperrt – Status nicht aktualisiert", path)
    return False
