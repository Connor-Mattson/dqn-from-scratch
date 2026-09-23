"""Pygame drawing for the maximization-bias casino.

Two entry points:

- ``CasinoRenderer.draw_env(env)`` — what ``env.render()`` returns: the room, the agent, the
  last payout. It knows nothing about any agent's values.
- ``render_duel(arms, config)`` — the side-by-side training replay behind
  ``results/casino/casino_duel.gif``. Each panel paints one agent's Q-values onto the
  machines, marks the value its target rule bootstraps from, and plays one greedy episode
  per checkpoint, so the viewer watches vanilla DQN talk itself into gambling.

The scene is top-down: the lobby on the left with a green exit, a neon entrance in the
dividing wall (every door leads to the same floor, so they share one entrance), and the
slot machines on the right.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from dqn.casino_env import EXIT, FLOOR, LOBBY  # noqa: E402

if TYPE_CHECKING:
    from dqn.casino_env import MaxBiasCasinoEnv
    from dqn.casino_experiment import ArmResult, CasinoConfig

BACKDROP = (18, 9, 14)
WALL = (46, 12, 24)
WALL_EDGE = (120, 86, 36)
CARPET = (84, 16, 34)
CARPET_PATTERN = (104, 26, 44)
LOBBY_TILE = (40, 30, 34)
LOBBY_TILE_ALT = (48, 37, 41)
GOLD = (236, 186, 80)
GOLD_DIM = (140, 104, 44)
NEON = (255, 72, 150)
EXIT_GREEN = (52, 211, 120)
WIN = (96, 226, 132)
LOSS = (246, 96, 96)
TEXT = (246, 238, 224)
TEXT_DIM = (178, 160, 150)
PANEL = (28, 16, 22)
MACHINE = (150, 24, 40)
MACHINE_TOP = (190, 40, 56)
SCREEN = (16, 12, 14)
# The plot palette's blue/orange, so the GIF and casino_bias.png agree on who is who.
ARM_COLORS = ((72, 150, 240), (240, 124, 70))

WIDTH = 600
HEADER_H = 54
SCENE_H = 270
CHART_H = 176
FOOTER_H = 44
PANEL_H = HEADER_H + SCENE_H + CHART_H + FOOTER_H
LOBBY_W = 190
WALL_W = 14
TITLE_H = 46
FRAMES_PER_EPISODE = 10


def _font(size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont("avenirnext,helveticaneue,helvetica,arial", size, bold=bold)


def _text(surface, text: str, pos, size: int = 16, color=TEXT, bold=False, anchor: str = "topleft") -> pygame.Rect:
    image = _font(size, bold).render(text, True, color)
    rect = image.get_rect(**{anchor: pos})
    surface.blit(image, rect)
    return rect


def _lerp(a, b, t: float):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _signed(value: float) -> str:
    """``+0.12`` / ``-0.12``, without the ``-0.00`` that rounding a tiny negative produces."""
    return f"{value + 0.0:+.2f}" if abs(value) >= 0.005 else "0.00"


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def value_color(value: float, limit: float) -> tuple[int, int, int]:
    """Red for 'this machine loses', green for 'this machine pays', grey near zero."""
    t = max(-1.0, min(1.0, value / limit))
    neutral = (120, 110, 110)
    end = WIN if t > 0 else LOSS
    t = abs(t) ** 0.7
    return tuple(int(n + (e - n) * t) for n, e in zip(neutral, end))


class CasinoRenderer:
    def __init__(self, n_actions: int, bet_mean: float, human: bool = False) -> None:
        pygame.init()
        pygame.font.init()
        self.n_actions = n_actions
        self.bet_mean = bet_mean
        self.human = human
        self.window = None
        self.clock = None
        self.machines = self._layout_machines()
        self.scene_rect = pygame.Rect(0, 0, WIDTH, SCENE_H)
        self.exit_rect = pygame.Rect(0, SCENE_H // 2 - 42, 16, 84)
        self.entrance_rect = pygame.Rect(LOBBY_W, SCENE_H // 2 - 46, WALL_W, 92)
        self.start_xy = (LOBBY_W * 0.55, SCENE_H / 2)
        self.exit_xy = (30, SCENE_H / 2)
        self.entrance_xy = (LOBBY_W + WALL_W + 18, SCENE_H / 2)

    # ------------------------------------------------------------------ layout
    def _layout_machines(self) -> list[pygame.Rect]:
        left, top = LOBBY_W + WALL_W + 44, 30
        area_w, area_h = WIDTH - left - 14, SCENE_H - top - 8

        def score(cols: int) -> float:  # fewest empty cells first, then cells shaped like a machine
            rows = math.ceil(self.n_actions / cols)
            return rows * cols - self.n_actions + abs(math.log((area_w / cols) / (area_h / rows) / 1.3))

        cols = min(range(1, self.n_actions + 1), key=score)
        rows = math.ceil(self.n_actions / cols)
        cell_w, cell_h = area_w / cols, area_h / rows
        w, h = min(64, cell_w - 8), min(50, cell_h - 8)
        rects = []
        for index in range(self.n_actions):
            row, col = divmod(index, cols)
            cx, cy = left + (col + 0.5) * cell_w, top + (row + 0.5) * cell_h
            rects.append(pygame.Rect(round(cx - w / 2), round(cy - h / 2), round(w), round(h)))
        return rects

    def machine_xy(self, index: int) -> tuple[float, float]:
        rect = self.machines[index]
        return (rect.centerx, rect.bottom + 6)

    # ------------------------------------------------------------------ scene
    def draw_scene(
        self,
        surface: pygame.Surface,
        agent_xy: tuple[float, float],
        lit_machine: int | None = None,
        payout: float | None = None,
        payout_t: float = 1.0,
        q_floor: np.ndarray | None = None,
        value_limit: float = 0.5,
        exit_lit: bool = False,
        entrance_lit: bool = False,
    ) -> None:
        # Lobby: checker tiles. Floor: carpet with a diamond pattern.
        for y in range(0, SCENE_H, 22):
            for x in range(0, LOBBY_W, 22):
                color = LOBBY_TILE if (x // 22 + y // 22) % 2 else LOBBY_TILE_ALT
                pygame.draw.rect(surface, color, (x, y, 22, 22))
        pygame.draw.rect(surface, CARPET, (LOBBY_W, 0, WIDTH - LOBBY_W, SCENE_H))
        for y in range(-20, SCENE_H + 20, 28):
            for x in range(LOBBY_W, WIDTH + 20, 28):
                pts = [(x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)]
                pygame.draw.polygon(surface, CARPET_PATTERN, pts, 1)
        _text(surface, "LOBBY", (LOBBY_W // 2 + 4, 14), 12, TEXT_DIM, bold=True, anchor="center")
        _text(surface, "CASINO FLOOR", (LOBBY_W + WALL_W + 12, 14), 12, GOLD_DIM, bold=True, anchor="midleft")

        # Dividing wall with the neon casino entrance.
        pygame.draw.rect(surface, WALL, (LOBBY_W, 0, WALL_W, SCENE_H))
        pygame.draw.line(surface, WALL_EDGE, (LOBBY_W, 0), (LOBBY_W, SCENE_H), 2)
        pygame.draw.line(surface, WALL_EDGE, (LOBBY_W + WALL_W, 0), (LOBBY_W + WALL_W, SCENE_H), 2)
        glow = NEON if entrance_lit else (150, 40, 90)
        pygame.draw.rect(surface, CARPET, self.entrance_rect)
        pygame.draw.rect(surface, glow, self.entrance_rect.inflate(4, 0), 3)
        sign = pygame.Surface((96, 20), pygame.SRCALPHA)
        _text(sign, "CASINO", (48, 10), 15, glow, bold=True, anchor="center")
        sign = pygame.transform.rotate(sign, 90)
        surface.blit(sign, sign.get_rect(midright=(LOBBY_W - 4, SCENE_H // 2)))
        _text(surface, f"{self.n_actions - 1} doors", (LOBBY_W - 26, self.entrance_rect.bottom + 16), 11, TEXT_DIM,
              anchor="center")

        # Exit door on the lobby's left wall.
        pygame.draw.rect(surface, WALL, (0, 0, 8, SCENE_H))
        green = EXIT_GREEN if exit_lit else (30, 130, 76)
        pygame.draw.rect(surface, green, self.exit_rect, border_radius=3)
        pygame.draw.rect(surface, (220, 255, 230) if exit_lit else (90, 180, 120), self.exit_rect, 2, border_radius=3)
        _text(surface, "EXIT", (24, self.exit_rect.top - 12), 13, green, bold=True, anchor="midleft")
        _text(surface, "$0, guaranteed", (24, self.exit_rect.bottom + 12), 11, TEXT_DIM, anchor="midleft")

        for index, rect in enumerate(self.machines):
            self._draw_machine(surface, index, rect, index == lit_machine, q_floor, value_limit)

        if payout is not None and lit_machine is not None:
            rect = self.machines[lit_machine]
            rise = 18 * payout_t
            color = WIN if payout > 0 else LOSS
            label = f"{'+' if payout >= 0 else '−'}${abs(payout):.2f}"
            box = _text(surface, label, (rect.centerx, rect.top - 10 - rise), 20, color, bold=True, anchor="center")
            pygame.draw.rect(surface, BACKDROP, box.inflate(10, 4), border_radius=6)
            pygame.draw.rect(surface, color, box.inflate(10, 4), 2, border_radius=6)
            _text(surface, label, box.center, 20, color, bold=True, anchor="center")

        self._draw_agent(surface, agent_xy)

    def _draw_machine(self, surface, index, rect, lit, q_floor, value_limit) -> None:
        shadow = rect.move(3, 3)
        pygame.draw.rect(surface, (40, 6, 14), shadow, border_radius=6)
        pygame.draw.rect(surface, MACHINE_TOP if lit else MACHINE, rect, border_radius=6)
        pygame.draw.rect(surface, GOLD if lit else GOLD_DIM, rect, 2, border_radius=6)
        screen = rect.inflate(-10, -rect.height * 0.45)
        screen.top = rect.top + 5
        pygame.draw.rect(surface, SCREEN, screen, border_radius=3)
        size = max(10, min(15, rect.height // 3))
        if q_floor is None:
            reels = "7 7 7" if lit else "$ ? $"
            _text(surface, reels, screen.center, size, GOLD if lit else TEXT_DIM, bold=True, anchor="center")
        else:
            value = float(q_floor[index])
            _text(surface, _signed(value), screen.center, size, value_color(value, value_limit), bold=True,
                  anchor="center")
        # Lever.
        pygame.draw.line(surface, TEXT_DIM, (rect.right - 2, rect.centery), (rect.right + 5, rect.centery - 8), 2)
        pygame.draw.circle(surface, LOSS, (rect.right + 5, rect.centery - 9), 3)
        _text(surface, str(index), (rect.centerx, rect.bottom - 8), 10, TEXT_DIM, anchor="center")

    def _draw_agent(self, surface, xy) -> None:
        x, y = int(xy[0]), int(xy[1])
        pygame.draw.ellipse(surface, (10, 4, 8), (x - 14, y + 8, 30, 10))  # shadow
        pygame.draw.ellipse(surface, (236, 226, 204), (x - 15, y - 9, 30, 20))  # shoulders, seen from above
        pygame.draw.ellipse(surface, (120, 110, 96), (x - 15, y - 9, 30, 20), 1)
        pygame.draw.circle(surface, (22, 20, 28), (x, y), 10)  # top hat brim
        pygame.draw.circle(surface, (6, 6, 10), (x, y), 6)
        pygame.draw.circle(surface, NEON, (x, y), 6, 2)  # hat band

    # ------------------------------------------------------------------ env.render()
    def draw_env(self, env: MaxBiasCasinoEnv) -> np.ndarray:
        surface = pygame.Surface((WIDTH, SCENE_H + FOOTER_H + 30))
        surface.fill(BACKDROP)
        scene = surface.subsurface(self.scene_rect)
        lit_machine = payout = None
        exit_lit = entrance_lit = False
        if env.room == LOBBY and env.done:  # just walked out
            agent_xy, exit_lit = self.exit_xy, True
        elif env.room == FLOOR and not env.done:  # just came through a door
            agent_xy, entrance_lit = self.entrance_xy, True
        elif env.last_room == FLOOR:  # just pulled a lever
            lit_machine, payout = env.last_action, env.last_reward
            agent_xy = self.machine_xy(lit_machine)
        else:
            agent_xy = self.start_xy
        self.draw_scene(scene, agent_xy, lit_machine, payout, exit_lit=exit_lit, entrance_lit=entrance_lit)

        bar = pygame.Rect(0, SCENE_H, WIDTH, surface.get_height() - SCENE_H)
        pygame.draw.rect(surface, PANEL, bar)
        pygame.draw.line(surface, GOLD_DIM, bar.topleft, bar.topright, 2)
        _text(surface, f"Every machine pays N({env.bet_mean:+.2f}, {env.sigma:g}²): it loses on average.",
              (14, bar.top + 12), 14, TEXT_DIM)
        bank_color = WIN if env.bankroll >= 0 else LOSS
        _text(surface, f"Bankroll {'+' if env.bankroll >= 0 else '−'}${abs(env.bankroll):.2f}",
              (WIDTH - 14, bar.top + 12), 15, bank_color, bold=True, anchor="topright")
        status = {None: "In the lobby. Walk out, or try a casino door?"}.get(env.last_action, "")
        if env.done and env.last_room == LOBBY:
            status = "Walked away with $0. The optimal move."
        elif env.room == FLOOR and not env.done:
            status = "On the casino floor. Every lever is a losing bet."
        elif env.done:
            status = f"Pulled machine {env.last_action}."
        _text(surface, status, (14, bar.top + 38), 15, TEXT)
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)).copy()

    def show(self, frame: np.ndarray, fps: int) -> None:
        if self.window is None:
            pygame.display.init()
            self.window = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
            pygame.display.set_caption("Maximization-bias casino")
            self.clock = pygame.time.Clock()
        pygame.event.pump()
        self.window.blit(pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), (0, 0))
        pygame.display.flip()
        self.clock.tick(fps)

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            self.window = None

    # ------------------------------------------------------------------ duel panel
    def draw_panel(
        self,
        label: str,
        color: tuple[int, int, int],
        step: int,
        q_lobby: np.ndarray,
        q_floor: np.ndarray,
        bootstrap: float,
        gamma: float,
        seeds_gambling: tuple[int, int],
        expected_loss: float,
        episode: dict,
        t: float,
        value_limit: float,
        double: bool,
    ) -> pygame.Surface:
        panel = pygame.Surface((WIDTH, PANEL_H))
        panel.fill(BACKDROP)

        # Header.
        _text(panel, label.upper(), (14, 10), 22, color, bold=True)
        _text(panel, f"training step {step:,}", (14, 36), 12, TEXT_DIM)
        bank_color = LOSS if expected_loss > 0 else TEXT_DIM
        _text(panel, f"expected losses −${expected_loss:.2f}", (WIDTH - 14, 12), 15, bank_color, bold=True,
              anchor="topright")
        _text(panel, f"{-self.bet_mean * 100:.0f}¢ per greedy gamble, one episode per checkpoint", (WIDTH - 14, 36), 12,
              TEXT_DIM, anchor="topright")

        # Scene with this agent's episode.
        scene = panel.subsurface(pygame.Rect(0, HEADER_H, WIDTH, SCENE_H))
        agent_xy, lit, payout, payout_t, exit_lit, entrance_lit = self._episode_pose(episode, t)
        self.draw_scene(scene, agent_xy, lit, payout, payout_t, q_floor, value_limit, exit_lit, entrance_lit)
        best = int(q_floor.argmax())
        crown = self.machines[best]
        pygame.draw.rect(scene, GOLD, crown.inflate(8, 8), 2, border_radius=8)
        _text(scene, "max", (crown.centerx, crown.top - 4), 11, GOLD, bold=True, anchor="midbottom")

        self._draw_chart(panel, pygame.Rect(0, HEADER_H + SCENE_H, WIDTH, CHART_H), q_lobby, q_floor, bootstrap,
                         gamma, value_limit, double)

        # Footer verdict.
        footer = pygame.Rect(0, PANEL_H - FOOTER_H, WIDTH, FOOTER_H)
        pygame.draw.rect(panel, PANEL, footer)
        gambles = int(q_lobby.argmax()) != EXIT
        verdict, verdict_color = ("GAMBLES", LOSS) if gambles else ("WALKS AWAY", WIN)
        rect = _text(panel, "Greedy policy:", (14, footer.centery), 16, TEXT_DIM, anchor="midleft")
        _text(panel, verdict, (rect.right + 8, footer.centery), 18, verdict_color, bold=True, anchor="midleft")
        gambling, seeds = seeds_gambling
        _text(panel, f"{gambling}/{seeds} seeds gambling right now", (WIDTH - 14, footer.centery), 15,
              LOSS if gambling * 2 > seeds else TEXT, bold=True, anchor="midright")
        return panel

    def _draw_chart(self, panel, area: pygame.Rect, q_lobby, q_floor, bootstrap, gamma, limit, double) -> None:
        pygame.draw.rect(panel, PANEL, area)
        pygame.draw.line(panel, GOLD_DIM, area.topleft, area.topright, 2)
        top, bottom = area.top + 34, area.bottom - 22
        zero_y = (top + bottom) / 2

        def y_of(value: float) -> float:
            return zero_y - (value / limit) * (bottom - top) / 2

        # Left: the decision in the lobby.
        left = pygame.Rect(area.left + 14, top, 150, bottom - top)
        _text(panel, "Lobby: exit vs best door", (left.left, area.top + 10), 13, TEXT, bold=True)
        door = float(q_lobby[1:].max())
        bars = [("exit", float(q_lobby[EXIT]), 0.0, EXIT_GREEN), ("door", door, gamma * self.bet_mean, NEON)]
        for i, (name, value, truth, bar_color) in enumerate(bars):
            x = left.left + 18 + i * 66
            y = y_of(value)
            pygame.draw.rect(panel, bar_color, pygame.Rect(x, min(y, zero_y), 38, max(2, abs(y - zero_y))),
                             border_radius=2)
            ty = y_of(truth)
            pygame.draw.line(panel, TEXT, (x - 5, ty), (x + 43, ty), 2)
            _text(panel, _signed(value), (x + 19, min(y, zero_y) - 3), 12, TEXT, bold=True, anchor="midbottom")
            _text(panel, name, (x + 19, bottom + 10), 12, TEXT_DIM, anchor="center")
        pygame.draw.line(panel, TEXT_DIM, (left.left, zero_y), (left.right, zero_y), 1)

        # Right: every machine's estimate, the truth, and the value that gets bootstrapped.
        right = pygame.Rect(area.left + 190, top, area.width - 204, bottom - top)
        title = "Floor: Q per machine" + (" (target prices the online pick)" if double else " (target takes the max)")
        _text(panel, title, (right.left, area.top + 10), 13, TEXT, bold=True)
        n = len(q_floor)
        slot = right.width / n
        best = int(q_floor.argmax())
        for i, value in enumerate(q_floor):
            y = y_of(float(value))
            x = right.left + i * slot + slot * 0.18
            c = value_color(float(value), limit)
            pygame.draw.rect(panel, c, pygame.Rect(x, min(y, zero_y), slot * 0.64, max(2, abs(y - zero_y))))
            if i == best:
                pygame.draw.rect(panel, GOLD, pygame.Rect(x - 2, min(y, zero_y) - 2, slot * 0.64 + 4,
                                                          max(2, abs(y - zero_y)) + 4), 1)
        pygame.draw.line(panel, TEXT_DIM, (right.left, zero_y), (right.right, zero_y), 1)
        truth_y = y_of(self.bet_mean)
        for x in range(right.left, right.right, 8):
            pygame.draw.line(panel, TEXT, (x, truth_y), (x + 4, truth_y), 1)
        _text(panel, f"truth {self.bet_mean:+.2f}", (right.right, truth_y + 3), 11, TEXT, anchor="topright")
        boot_y = y_of(bootstrap)
        pygame.draw.line(panel, GOLD, (right.left, boot_y), (right.right, boot_y), 2)
        _text(panel, f"bootstrapped into every door: {_signed(bootstrap)}", (right.left, boot_y - 3), 12, GOLD, bold=True,
              anchor="bottomleft")
        _text(panel, "0", (right.left - 4, zero_y), 10, TEXT_DIM, anchor="midright")

    def _episode_pose(self, episode: dict, t: float):
        """Agent position and highlights ``t`` of the way (0..1) through one greedy episode."""
        if not episode["gambles"]:
            walk = _ease(min(1.0, t / 0.6))
            return _lerp(self.start_xy, self.exit_xy, walk), None, None, 1.0, t >= 0.6, False
        machine = episode["machine"]
        if t < 0.35:
            return _lerp(self.start_xy, self.entrance_xy, _ease(t / 0.35)), None, None, 1.0, False, True
        if t < 0.65:
            xy = _lerp(self.entrance_xy, self.machine_xy(machine), _ease((t - 0.35) / 0.3))
            return xy, None, None, 1.0, False, False
        return self.machine_xy(machine), machine, episode["payout"], (t - 0.65) / 0.35, False, False


def render_duel(arms: Sequence[ArmResult], config: CasinoConfig, seed_index: int | None = None,
                checkpoint_stride: int = 2) -> list[np.ndarray]:
    """Frames of the two arms side by side over training, one greedy episode per checkpoint.

    Both panels replay the same seed: the one whose gap in gamble rate between the arms is
    the median across seeds, so the replay is typical rather than cherry-picked.

    Payouts are real draws, but the header tallies *expected* losses (``-bet_mean`` per
    gamble). With sigma = 2 against a 10-cent edge, a realised bankroll over a few dozen
    episodes is mostly luck and could rank the arms either way.
    """
    renderer = CasinoRenderer(config.n_actions, config.bet_mean)
    if seed_index is None:
        gaps = np.array([np.mean(a.gambles) - np.mean(b.gambles) for a, b in zip(arms[0].runs, arms[1].runs)])
        seed_index = int(np.argsort(gaps)[len(gaps) // 2])
    runs = [arm.runs[seed_index] for arm in arms]
    checkpoints = list(range(0, len(runs[0].steps), checkpoint_stride))
    values = np.concatenate([np.abs(np.concatenate([run.q_floor, run.q_lobby])).ravel() for run in runs])
    value_limit = max(0.3, float(np.percentile(values, 98)))

    # Pre-roll each arm's greedy episodes with one payout stream, so both face the same luck.
    luck = np.random.default_rng(1234)
    payouts = luck.normal(config.bet_mean, config.sigma, size=len(runs[0].steps))
    episodes, losses = [], []
    for run in runs:
        loss, arm_episodes, arm_losses = 0.0, [], []
        for k in checkpoints:
            gambles = int(run.q_lobby[k].argmax()) != EXIT
            arm_episodes.append({"gambles": gambles, "machine": int(run.q_floor[k].argmax()), "payout": float(payouts[k])})
            arm_losses.append(loss)  # expected losses before this episode
            loss += -config.bet_mean if gambles else 0.0
        episodes.append(arm_episodes)
        losses.append(arm_losses)

    total_w, total_h = 2 * WIDTH + 12, TITLE_H + PANEL_H
    frames = []
    for e, k in enumerate(checkpoints):
        seeds = len(arms[0].runs)
        gambling = [int(sum(run.gambles[k] for run in arm.runs)) for arm in arms]
        for f in range(FRAMES_PER_EPISODE):
            t = f / (FRAMES_PER_EPISODE - 1)
            canvas = pygame.Surface((total_w, total_h))
            canvas.fill(BACKDROP)
            _text(canvas, f"{config.n_actions} slot machines, each paying N({config.bet_mean:+.1f}, {config.sigma:g}²). "
                          "Walking out is worth $0. Which agent sees through the noise?",
                  (total_w // 2, TITLE_H // 2), 17, TEXT, bold=True, anchor="center")
            for i, (arm, run) in enumerate(zip(arms, runs)):
                episode = episodes[i][e]
                loss = losses[i][e] + (-config.bet_mean if episode["gambles"] and t >= 0.65 else 0.0)
                panel = renderer.draw_panel(
                    arm.label, ARM_COLORS[i], run.steps[k], run.q_lobby[k], run.q_floor[k],
                    run.bootstrap_bias[k] + config.bet_mean, config.gamma, (gambling[i], seeds), loss,
                    episode, t, value_limit, arm.double,
                )
                canvas.blit(panel, (i * (WIDTH + 12), TITLE_H))
            frames.append(np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2)).copy())
        frames.extend([frames[-1]] * 3)  # a beat to read the numbers before the next checkpoint
    return frames
