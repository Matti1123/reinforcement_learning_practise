"""Lokaler MCP-Server zum Auswerten gespeicherter CartPole-Policies."""

from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
SIMPLE_DIR = PROJECT_DIR / "cartpole_simple"
sys.path.insert(0, str(SIMPLE_DIR))

from mcp.server.fastmcp import FastMCP

from policy_testing import test_policy

POLICY_DIR = SIMPLE_DIR / "policies"
server = FastMCP("CartPole Policy Testing")


@server.tool()
def policy_testen(dateiname: str, episoden: int = 50, seed: int = 10000) -> dict:
    """Testet eine .pt-Datei aus cartpole_simple/policies/; liefert Testkennzahlen.

    dateiname ist der Dateiname ohne Ordner. episoden liegt zwischen 1 und 500.
    Die Seeds beginnen bei seed und steigen pro Episode um eins.
    """
    if not 1 <= episoden <= 500 or seed < 0:
        raise ValueError("episoden muss zwischen 1 und 500 und seed nicht negativ sein")
    path = (POLICY_DIR / dateiname).resolve()
    if path.parent != POLICY_DIR or path.suffix != ".pt":
        raise ValueError("Nur .pt-Dateien direkt im policies-Ordner sind erlaubt")
    return test_policy(path, episodes=episoden, seed=seed)


if __name__ == "__main__":
    server.run(transport="stdio")
