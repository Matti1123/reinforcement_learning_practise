# CartPole als MCP-Tool

Das lokale Tool `policy_testen` testet eine Policy aus `cartpole_simple/policies/`. Es gibt
unter anderem Erfolgsrate, mittlere Dauer, Abstand zur Mitte und Winkel aus.
Es trainiert keine Policy und verändert keine Dateien.

Das MCP-SDK in der bestehenden virtuellen Umgebung installieren:

```bash
.venv/bin/python -m pip install 'mcp>=1,<2'
```

In einem MCP-fähigen Agenten den Server als lokalen stdio-Prozess konfigurieren:
Als Programm den **absoluten Pfad** zu `.venv/bin/python` und als Argument den
**absoluten Pfad** zu `mcp/mcp_server.py` angeben. Die genaue Konfiguration hängt
vom verwendeten Agenten ab. Der Agent startet den Server selbst; direkt im
Terminal gestartet wartet er auf MCP-Nachrichten.

Beispiel für dieses Projekt:

```text
Programm: /home/flets/reinforcement_learning_medtech/.venv/bin/python
Argument: /home/flets/reinforcement_learning_medtech/mcp/mcp_server.py
```

Eine bestehende MCP-Konfiguration mit dem alten Pfad muss entsprechend
aktualisiert und der Agent beziehungsweise die MCP-Verbindung neu gestartet werden.

Beispiel für einen Tool-Aufruf: Dateiname `policy_20261124_214753_d3e62823.pt`,
`episoden` 50 und `seed` 10000. Der Agent benötigt Zugriff auf den lokalen
Ordner `cartpole_simple/policies/`, dessen Dateien nicht in Git gespeichert werden.
Das Tool unterstützt bisher nur `cartpole_simple`. Nach Änderungen am Server
die MCP-Verbindung neu starten, damit der neue Code geladen wird.
