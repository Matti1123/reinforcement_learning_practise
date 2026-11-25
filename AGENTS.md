# Zusammenarbeit im Projekt

Bei ausdrücklich angeforderter Arbeit mit Subagenten koordiniert der Hauptagent die Aufgaben und fasst die Ergebnisse zusammen. Der Researchagent untersucht Policy- und Trainingsansätze und führt beauftragte Trainingsläufe durch. Der Testagent bewertet gespeicherte Policies mit dem MCP-Werkzeug `cartpole.policy_testen` und dokumentiert die Testbedingungen. Gemeinsame Dateiänderungen werden koordiniert; Vergleiche verwenden gleiche Testbedingungen und möglichst von den Trainings-Seeds getrennte Test-Seeds.
