"""Wagen mit zwei Stäben und einer CartPole-Policy sichtbar simulieren."""

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import torch

PROJEKT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJEKT))
from modelle.policy import Policy
from modelle.gelenk_policy import GelenkPolicy, beobachtung


class Doppelpendel:
    """Ebenes Doppelpendel auf einem horizontal beweglichen Wagen.

    Winkel werden von der aufrechten Senkrechten aus gemessen. Die Stäbe sind
    homogene starre Körper; beide Gelenke sind frei drehbar.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.dt = 1 / 240
        self.gravitation = 9.81
        self.wagenmasse = 1.0
        self.massen = (0.3, 0.2)
        self.laengen = (1.0, 0.8)
        self.zustand = np.array([0.0, 0.05, -0.05, 0.0, 0.0, 0.0])

    def schritt(self, aktion):
        x, a, b, vx, va, vb = self.zustand
        m1, m2 = self.massen
        l1, l2 = self.laengen
        g = self.gravitation
        # Schwerpunktabstände und Trägheitsmomente der homogenen Stäbe.
        r1, r2 = l1 / 2, l2 / 2
        i1, i2 = m1 * l1**2 / 12, m2 * l2**2 / 12
        k1 = m1 * r1 + m2 * l1
        k2 = m2 * r2
        k12 = m2 * l1 * r2
        matrix = np.array([
            [self.wagenmasse + m1 + m2, k1 * math.cos(a), k2 * math.cos(b)],
            [k1 * math.cos(a), i1 + m1 * r1**2 + m2 * l1**2, k12 * math.cos(a - b)],
            [k2 * math.cos(b), k12 * math.cos(a - b), i2 + m2 * r2**2],
        ])
        kraft = np.array([
            10.0 * aktion - 0.1 * vx + k1 * math.sin(a) * va**2 + k2 * math.sin(b) * vb**2,
            k1 * g * math.sin(a) - k12 * math.sin(a - b) * vb**2 - 0.02 * va,
            k2 * g * math.sin(b) + k12 * math.sin(a - b) * va**2 - 0.02 * vb,
        ])
        beschleunigung = np.linalg.solve(matrix, kraft)
        geschwindigkeit = self.zustand[3:] + self.dt * beschleunigung
        position = self.zustand[:3] + self.dt * geschwindigkeit
        # Elastischer Anschlag hält den Wagen auf der sichtbaren Schiene.
        if abs(position[0]) > 2.4:
            position[0] = np.clip(position[0], -2.4, 2.4)
            geschwindigkeit[0] *= -0.5
        self.zustand = np.concatenate((position, geschwindigkeit))

    def zufallsaktion(self):
        return int(self.rng.choice((-1, 1)))


def main():
    parser = argparse.ArgumentParser(description="CartPole mit zwei Stäben und Policy-Steuerung")
    parser.add_argument("--seed", type=int, default=None, help="Seed für reproduzierbare Zufallsaktionen")
    parser.add_argument("--headless", action="store_true", help="Ohne Fenster simulieren")
    parser.add_argument("--steps", type=int, default=None, help="Anzahl Simulationsschritte (240 pro Sekunde)")
    parser.add_argument("--steuerung", choices=("policy", "zufall"), default="policy")
    parser.add_argument("--policy", type=Path, default=PROJEKT / "policies/cartpole_simple_mit_mitten_reward.pt", help="Pfad zur Simple- oder Gelenk-Policy")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps muss mindestens 1 sein")
    if args.headless and args.steps is None:
        parser.error("--headless benötigt --steps")

    simulation = Doppelpendel(args.seed)
    policy = None
    if args.steuerung == "policy":
        try:
            gewichte = torch.load(args.policy, map_location="cpu", weights_only=True)
            eingaben = gewichte["network.0.weight"].shape[1]
            if eingaben not in (4, 6):
                raise ValueError(f"Nicht unterstützte Eingabegröße: {eingaben}")
            policy = Policy() if eingaben == 4 else GelenkPolicy()
            policy.load_state_dict(gewichte)
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            parser.error(f"Policy konnte nicht geladen werden: {error}")
        policy.eval()

    def naechste_aktion():
        if policy is None:
            return simulation.zufallsaktion()
        if isinstance(policy, GelenkPolicy):
            with torch.no_grad():
                return 1 if policy(beobachtung(simulation.zustand)).probs.argmax().item() else -1
        x, winkel, _, vx, winkelgeschwindigkeit, _ = simulation.zustand
        # CartPole-v1 erwartet Winkel im Bereich um die aufrechte Stellung.
        winkel = (winkel + math.pi) % (2 * math.pi) - math.pi
        with torch.no_grad():
            return 1 if policy([x, vx, winkel, winkelgeschwindigkeit]).probs.argmax().item() else -1

    aktionsintervall = 5 if policy is not None else 48
    if args.headless:
        for schritt in range(args.steps):
            if schritt % aktionsintervall == 0:
                aktion = naechste_aktion()
            simulation.schritt(aktion)
        print("Endzustand [x, Winkel 1, Winkel 2, Geschwindigkeiten]:", simulation.zustand)
        return

    import pygame

    pygame.init()
    fenster = pygame.display.set_mode((1000, 650))
    pygame.display.set_caption("CartPole mit Gelenk – Steuerung")
    uhr = pygame.time.Clock()
    schrift = pygame.font.Font(None, 26)
    schritt = 0
    aktion = naechste_aktion()
    laeuft = True
    try:
        while laeuft and (args.steps is None or schritt < args.steps):
            for ereignis in pygame.event.get():
                if ereignis.type == pygame.QUIT or (ereignis.type == pygame.KEYDOWN and ereignis.key == pygame.K_ESCAPE):
                    laeuft = False
            # Vier Integrationsschritte pro Bild entsprechen 60 Bildern pro Sekunde.
            for _ in range(4):
                if args.steps is not None and schritt >= args.steps:
                    break
                if schritt % aktionsintervall == 0:
                    aktion = naechste_aktion()
                simulation.schritt(aktion)
                schritt += 1

            x, a, b = simulation.zustand[:3]
            skala, boden = 150, 400
            basis = (500 + x * skala, boden - 25)
            gelenk = (basis[0] + simulation.laengen[0] * skala * math.sin(a),
                      basis[1] - simulation.laengen[0] * skala * math.cos(a))
            spitze = (gelenk[0] + simulation.laengen[1] * skala * math.sin(b),
                      gelenk[1] - simulation.laengen[1] * skala * math.cos(b))
            fenster.fill((245, 247, 250))
            pygame.draw.line(fenster, (110, 120, 130), (140, boden), (860, boden), 4)
            pygame.draw.rect(fenster, (45, 105, 180), (basis[0] - 38, boden - 25, 76, 25), border_radius=5)
            pygame.draw.line(fenster, (225, 130, 35), basis, gelenk, 10)
            pygame.draw.line(fenster, (180, 65, 70), gelenk, spitze, 9)
            for punkt in (basis, gelenk, spitze):
                pygame.draw.circle(fenster, (35, 40, 50), (round(punkt[0]), round(punkt[1])), 6)
            text = f"{args.steuerung.title()}: {'rechts' if aktion > 0 else 'links'}   |   ESC: beenden"
            fenster.blit(schrift.render(text, True, (35, 40, 50)), (25, 20))
            pygame.display.flip()
            uhr.tick(60)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
