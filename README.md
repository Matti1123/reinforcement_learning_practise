# Reinforcement-Learning-Experimente

- `cartpole_simple/`: bestehende CartPole-v1-Experimente, Policies und Ausgaben.
- `cartpole_gelenk/`: Platz für zukünftige Experimente mit einem Gelenk.
- `mcp/`: gemeinsamer MCP-Server; der Policy-Test unterstützt bisher nur `cartpole_simple`.
- `.codex/` und `AGENTS.md`: projektweite Agenten und Konfiguration.

Das bestehende Training lässt sich vom Projektordner aus starten:

```bash
.venv/bin/python cartpole_simple/cartpole.py --no-render
```

Policies und Lernkurven werden unabhängig vom aktuellen Arbeitsverzeichnis in
`cartpole_simple/policies/` beziehungsweise `cartpole_simple/outputs/` gespeichert.
