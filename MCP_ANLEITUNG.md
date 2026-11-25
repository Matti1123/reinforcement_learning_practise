# CartPole als MCP-Tool

Das lokale Tool `policy_testen` testet eine Policy aus `policies/`. Es gibt
unter anderem Erfolgsrate, mittlere Dauer, Abstand zur Mitte und Winkel aus.
Es trainiert keine Policy und verändert keine Dateien.

Das MCP-SDK in der bestehenden virtuellen Umgebung installieren:

```bash
.venv/bin/python -m pip install 'mcp>=1,<2'
```

In einem MCP-fähigen Agenten den Server als lokalen stdio-Prozess konfigurieren:
Als Programm den **absoluten Pfad** zu `.venv/bin/python` und als Argument den
**absoluten Pfad** zu `mcp_server.py` angeben. Die genaue Konfiguration hängt
vom verwendeten Agenten ab. Der Agent startet den Server selbst; direkt im
Terminal gestartet wartet er auf MCP-Nachrichten.

Beispiel für einen Tool-Aufruf: Dateiname `policy_20261124_214753_d3e62823.pt`,
`episoden` 50 und `seed` 10000. Der Agent benötigt Zugriff auf den lokalen
Ordner `policies/`, dessen Dateien nicht in Git gespeichert werden.
