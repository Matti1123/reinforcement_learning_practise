# CartPole mit Gelenk

Die Simulation zeigt einen Wagen mit zwei frei verbundenen Stäben. Der Wagen
erhält standardmäßig Aktionen der gespeicherten Simple-Policy mit Mitten-Reward.
Sie sieht nur Wagenposition, Wagengeschwindigkeit und den unteren Stab. Sie wurde
für das einfache CartPole trainiert und nicht für das Doppelpendel. Die
Simulation ist ein erster vereinfachter physikalischer Prototyp; eine
Gymnasium-Umgebung ist noch nicht implementiert.

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

## Training

```bash
.venv/bin/python cartpole_gelenk/cartpole_gelenk_training.py --episodes 1000
```

Das Training nutzt REINFORCE und eine Policy mit beiden Winkeln und beiden
Winkelgeschwindigkeiten. Der Reward belohnt beide aufrechten Stäbe und leicht
die Nähe zur Mitte. Eine Episode endet, wenn ein Stab über 0,5 Radiant geneigt
ist, der Wagen den Rand erreicht oder die maximale Schrittzahl erreicht wird.
Alle 50 Episoden wird auf zehn festen, vom Training getrennten Seeds bewertet;
die beste anhand des mittleren Rewards bewertete Policy dieses Laufs wird mit
einem eindeutigen Namen in `policies/` gespeichert. Die Auswahl auf denselben
Seeds ist eine Validierung, kein unabhängiger abschließender Test.
Eine Lernkurve mit den Rewards pro Trainingsepisode, ihrem gleitenden
Durchschnitt und dem Validierungs-Reward wird in `cartpole_gelenk/outputs/`
gespeichert und bei jeder Validierung aktualisiert. Der Dateiname gehört zum
selben Trainingslauf wie die gespeicherte Policy.

Eine gespeicherte Gelenk-Policy anzeigen:

```bash
.venv/bin/python cartpole_gelenk/simulation.py --policy policies/DATEINAME.pt
```
