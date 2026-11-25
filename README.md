# Reinforcement-Learning-Experimente

- `cartpole_simple/`: bestehende CartPole-v1-Experimente und Ausgaben.
- `cartpole_gelenk/`: Platz für zukünftige Experimente mit einem Gelenk.
- `modelle/`: gemeinsame Architektur der CartPole-Policy.
- `policies/`: gespeicherte Policies; nicht in Git enthalten.
- `mcp/`: gemeinsamer MCP-Server; der Policy-Test unterstützt bisher nur `cartpole_simple`.
- `.codex/` und `AGENTS.md`: projektweite Agenten und Konfiguration.

Das bestehende Training lässt sich vom Projektordner aus starten:

```bash
.venv/bin/python cartpole_simple/cartpole.py --no-render
```

Policies und Lernkurven werden unabhängig vom aktuellen Arbeitsverzeichnis in
`policies/` beziehungsweise `cartpole_simple/outputs/` gespeichert.

Die bisherige `policy_20261124_214753_d3e62823.pt` heißt jetzt
`cartpole_simple_mit_mitten_reward.pt`. Die andere gespeicherte Policy heißt
`cartpole_simple_ohne_mitten_reward.pt`; ihre Trainingshistorie ist hier nicht
separat dokumentiert.
