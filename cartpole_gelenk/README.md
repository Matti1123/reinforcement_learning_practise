# CartPole mit Gelenk

Die Simulation zeigt einen Wagen mit zwei frei verbundenen Stäben. Der Wagen
erhält standardmäßig Aktionen der gespeicherten Simple-Policy mit Mitten-Reward.
Sie sieht nur Wagenposition, Wagengeschwindigkeit und den unteren Stab. Sie wurde
für das einfache CartPole trainiert und nicht für das Doppelpendel. Die
Simulation ist ein erster vereinfachter physikalischer Prototyp; eine
Gymnasium-Umgebung und ein Training sind noch nicht implementiert.

Vom Projektordner aus starten:

```bash
.venv/bin/python cartpole_gelenk/simulation.py
```

Das Fenster mit Escape oder dem Schließen-Knopf beenden. Mit `--steuerung zufall`
werden stattdessen alle 0,2 Sekunden zufällige Aktionen ausgeführt; mit `--seed 42`
lassen sie sich reproduzieren. Eine andere Policy lässt sich mit `--policy PFAD`
auswählen. Ein kurzer Test ohne Fenster:

```bash
.venv/bin/python cartpole_gelenk/simulation.py --headless --steps 2400 --seed 42
```

Der vorhandene MCP-Test unterstützt derzeit nur `cartpole_simple`.
