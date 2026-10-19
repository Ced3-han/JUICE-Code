"""VSP maze navigation (FrozenLake) as a joint planning task (paper Sec. 5.4, App. D.4).

Task
----
A map is ``grid``: n strings over S / F / H / G (start, frozen, hole, goal), the
FrozenLake ``desc`` of the VSP benchmark (Wu et al., "VSP: Diagnosing the Dual
Challenges of Perception and Reasoning in Spatial Planning Tasks for MLLMs").
Input:  the map image and the question
        "FrozenLake {n}x{n}. Move the player to the goal without falling into a
        hole. Give the action plan (L, R, U, D)."
Output: the action plan "R,R,D" (text) and the map with a red route stroke (image).
Metric: the official VSP rule on the text plan: the player reaches the goal before
        it falls into a hole (gym FrozenLake-v1, not slippery). VSP levels 3 to 6
        are the benchmark; levels 7 and 8 are larger than any training map.

Canvas
------
224 px. One map cell is 28 px = 2 x 2 encoder patches. The n x n map is centered
at the whole-patch offset 14 * (8 - n) on white. The cell tiles are the 64 px
sprites of the VSP renders, resized to 28 px (LANCZOS). The sprites are not
shipped: load_sprites() cuts them out of the official VSP map images. (The VSP
renders use the gym FrozenLake art.)

The solved image adds a red stroke inside each FROZEN cell on the route (entry
edge -> center -> exit edge). Start, goal and hole cells look the same in the
input and in the solved image, so their patches are known at sampling time and
only the frozen-cell patches are generated.

Moves follow gym FrozenLake-v1 (is_slippery=False): a move off the map keeps the
player in place, a hole ends the episode in failure, the goal ends it in success,
and moves after the end are ignored.
"""

from __future__ import annotations

import hashlib
import io
import random
from collections import deque
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

SIZE = 224
PATCH = 14
GRID = SIZE // PATCH            # 16
CELL = 2 * PATCH                # 28 px per map cell
MAX_N = SIZE // CELL            # 8
SPRITE = 64                     # cell size of the VSP renders
ROUTE_RGB = (255, 0, 0)
STROKE_LO, STROKE_HI = 11, 16   # 6 px stroke band inside the 28 px cell
ROUTE_FRAC_THRESHOLD = 0.05     # red-pixel fraction that puts a cell on the route
LEVELS = (3, 4, 5, 6, 7, 8)     # official VSP maze levels (100 maps each)
BENCHMARK_LEVELS = (3, 4, 5, 6)

MOVES = {"L": (0, -1), "D": (1, 0), "R": (0, 1), "U": (-1, 0)}
CANON_ORDER = "LUDR"            # BFS neighbor order of the training answers
QUESTION_TMPL = ("FrozenLake {n}x{n}. Move the player to the goal without falling into a hole. "
                 "Give the action plan (L, R, U, D).")
MAX_ANSWER_TOKENS = 60


def question(n: int) -> str:
    return QUESTION_TMPL.format(n=n)


def answer_str(moves: Sequence[str]) -> str:
    return ",".join(moves)


def layout_hash(grid: Sequence[str]) -> str:
    return hashlib.sha1("/".join(grid).encode("ascii")).hexdigest()


def find(grid: Sequence[str], ch: str) -> tuple[int, int]:
    for i, row in enumerate(grid):
        j = row.find(ch)
        if j >= 0:
            return (i, j)
    raise ValueError(f"no {ch!r} in grid {grid}")


def table_to_grid(table_text: str) -> tuple[str, ...]:
    """VSP ``maps/level*/table/*.txt`` -> FrozenLake desc rows."""
    sym = {"_": "F", "#": "H", "@": "S", "*": "G"}
    rows = []
    for line in table_text.strip().split("\n")[1:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")][1:]
        rows.append("".join(sym[c] for c in cells))
    if any(len(r) != len(rows) for r in rows):
        raise ValueError("non-square VSP table")
    return tuple(rows)


# ----------------------------------------------------------------------------- solving


def _nbrs(grid, cell, order: str = CANON_ORDER):
    n = len(grid)
    for m in order:
        r, c = cell[0] + MOVES[m][0], cell[1] + MOVES[m][1]
        if 0 <= r < n and 0 <= c < n and grid[r][c] != "H":
            yield m, (r, c)


def bfs_route(grid: Sequence[str], order: str = CANON_ORDER):
    """Shortest route (cells from S to G, moves) by BFS in neighbor order `order`, or None."""
    s, g = find(grid, "S"), find(grid, "G")
    prev = {s: None}
    dq = deque([s])
    while dq:
        cur = dq.popleft()
        if cur == g:
            break
        for m, nxt in _nbrs(grid, cur, order):
            if nxt not in prev:
                prev[nxt] = (cur, m)
                dq.append(nxt)
    if g not in prev:
        return None
    cells, moves, cur = [g], [], g
    while prev[cur] is not None:
        cur, m = prev[cur]
        cells.append(cur)
        moves.append(m)
    return cells[::-1], moves[::-1]


def bfs_dist(grid: Sequence[str]) -> dict:
    """Hop distance from S over non-hole cells (G is not expanded)."""
    s, g = find(grid, "S"), find(grid, "G")
    dist = {s: 0}
    dq = deque([s])
    while dq:
        cur = dq.popleft()
        if cur == g:
            continue
        for _m, nxt in _nbrs(grid, cur):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                dq.append(nxt)
    return dist


def simulate(grid: Sequence[str], moves: Iterable[str]):
    """FrozenLake-v1 rollout (not slippery) -> (status, cells visited from S, moves used).

    status: "goal", "hole", "none" (plan ended first) or "invalid" (a token that is not L/R/U/D).
    """
    n = len(grid)
    cur = find(grid, "S")
    cells = [cur]
    for k, m in enumerate(moves):
        if m not in MOVES:
            return "invalid", cells, k
        r, c = cur[0] + MOVES[m][0], cur[1] + MOVES[m][1]
        if 0 <= r < n and 0 <= c < n:
            cur = (r, c)
        cells.append(cur)
        if grid[cur[0]][cur[1]] == "H":
            return "hole", cells, k + 1
        if grid[cur[0]][cur[1]] == "G":
            return "goal", cells, k + 1
    return "none", cells, len(cells) - 1


def parse_answer(text: str) -> list[str]:
    """Plan text -> tokens, as the official VSP eval.py parses it (split on ",", drop quotes, newlines, periods, spaces)."""
    out = []
    for tok in str(text).split(","):
        for ch in ('"', "'", "\n", "."):
            tok = tok.replace(ch, "")
        out.append(tok.strip())
    return out


# ----------------------------------------------------------------------------- geometry and rendering


def offset(n: int) -> int:
    if not 1 <= n <= MAX_N:
        raise ValueError(f"n={n} outside 1..{MAX_N}")
    return PATCH * (MAX_N - n)


def cell_xy(n: int, r: int, c: int) -> tuple[int, int]:
    off = offset(n)
    return off + CELL * c, off + CELL * r


def known_patch_flags(grid: Sequence[str]) -> list[int]:
    """256 ints in patch order (row-major): 1 = known (background, hole, start, goal), 0 = generated (frozen cells)."""
    n = len(grid)
    flags = [1] * (GRID * GRID)
    for r in range(n):
        for c in range(n):
            if grid[r][c] == "F":
                x, y = cell_xy(n, r, c)
                for dp in (0, 1):
                    for dq in (0, 1):
                        flags[(y // PATCH + dp) * GRID + (x // PATCH + dq)] = 0
    return flags


def load_sprites(maps_dir: str | Path) -> dict[str, np.ndarray]:
    """Cut the four 64 px tiles (S, F, H, G) out of the official VSP map images.

    maps_dir is ``<VSP repo>/maze/task-main/maps``. Every tile of a kind is the same
    in all VSP renders, so the first occurrence (levels 3..8, maps in id order) is used.
    """
    maps_dir = Path(maps_dir)
    sprites: dict[str, np.ndarray] = {}
    for level in LEVELS:
        for map_id in range(100):
            grid = table_to_grid((maps_dir / f"level{level}" / "table" / f"{map_id}.txt").read_text())
            missing = {ch for row in grid for ch in row} - set(sprites)
            if not missing:
                continue
            image = np.asarray(Image.open(maps_dir / f"level{level}" / "img" / f"{map_id}.png").convert("RGB"))
            for r, row in enumerate(grid):
                for c, ch in enumerate(row):
                    if ch in missing and ch not in sprites:
                        sprites[ch] = image[SPRITE * r:SPRITE * (r + 1), SPRITE * c:SPRITE * (c + 1)].copy()
            if len(sprites) == 4:
                return sprites
    raise RuntimeError(f"could not find all four tiles under {maps_dir}")


def cell_tiles(sprites: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """64 px sprites -> 28 px cell tiles (LANCZOS)."""
    return {k: np.asarray(Image.fromarray(v).resize((CELL, CELL), Image.LANCZOS), dtype=np.uint8) for k, v in sprites.items()}


def _stroke(arr: np.ndarray, x0: int, y0: int, sides: set) -> None:
    lo, hi = STROKE_LO, STROKE_HI
    arr[y0 + lo:y0 + hi + 1, x0 + lo:x0 + hi + 1] = ROUTE_RGB
    if "L" in sides:
        arr[y0 + lo:y0 + hi + 1, x0:x0 + hi + 1] = ROUTE_RGB
    if "R" in sides:
        arr[y0 + lo:y0 + hi + 1, x0 + lo:x0 + CELL] = ROUTE_RGB
    if "U" in sides:
        arr[y0:y0 + hi + 1, x0 + lo:x0 + hi + 1] = ROUTE_RGB
    if "D" in sides:
        arr[y0 + lo:y0 + CELL, x0 + lo:x0 + hi + 1] = ROUTE_RGB


def render(grid: Sequence[str], tiles: dict[str, np.ndarray], route_cells: Sequence[tuple[int, int]] | None = None) -> Image.Image:
    """224 px RGB image. `tiles` from cell_tiles(); route_cells (S .. G) draws the plan."""
    n = len(grid)
    arr = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
    for r in range(n):
        for c in range(n):
            x, y = cell_xy(n, r, c)
            arr[y:y + CELL, x:x + CELL] = tiles[grid[r][c]]
    side_of = {v: m for m, v in MOVES.items()}
    if route_cells:
        cells = [tuple(c) for c in route_cells]
        for k in range(1, len(cells) - 1):
            r, c = cells[k]
            if grid[r][c] != "F":
                raise ValueError(f"route cell {cells[k]} is {grid[r][c]!r}, not F")
            sides = {side_of[(o[0] - r, o[1] - c)] for o in (cells[k - 1], cells[k + 1])}
            _stroke(arr, *cell_xy(n, r, c), sides)
    return Image.fromarray(arr)


def route_cells_from_image(arr: np.ndarray, grid: Sequence[str]) -> set:
    """Image (float [H, W, 3] in [0, 1], or uint8) -> frozen cells whose red-pixel fraction is above the threshold."""
    a = np.asarray(arr, dtype=np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    n = len(grid)
    red = (a[..., 0] > 0.6) & (a[..., 1] < 0.35) & (a[..., 2] < 0.35)
    out = set()
    for r in range(n):
        for c in range(n):
            if grid[r][c] == "F":
                x, y = cell_xy(n, r, c)
                if red[y:y + CELL, x:x + CELL].mean() > ROUTE_FRAC_THRESHOLD:
                    out.add((r, c))
    return out


# ----------------------------------------------------------------------------- scoring


def score(grid: Sequence[str], gold_moves: Sequence[str], img_route: set, text: str) -> dict:
    """Scores of one (image route, text plan) pair.

    text_success = the official VSP rule (reach G before any H): the metric of the paper.
    img_valid = the red cells are the inner cells of SOME shortest route.
    joint_consistent = shortest text plan, valid image route, and the same cells.
    """
    grid = tuple(grid)
    s, g = find(grid, "S"), find(grid, "G")
    d_sg = bfs_dist(grid)[g]
    _status, gold_cells, _ = simulate(grid, gold_moves)
    gold_inner = {c for c in gold_cells if c not in (s, g)}

    toks = parse_answer(text)
    status, cells, used = simulate(grid, toks)
    out = {"n": len(grid), "shortest_len": d_sg,
           "text_parse_fail": len(toks) == 0 or any(t not in MOVES for t in toks),
           "text_success": status == "goal", "text_shortest": status == "goal" and used == d_sg}
    out["text_valid"] = out["text_success"]
    out["text_exact"] = toks == list(gold_moves)
    text_inner = {c for c in cells if c not in (s, g)}

    route = set(img_route)
    frozen = [(r, c) for r in range(len(grid)) for c in range(len(grid)) if grid[r][c] == "F"]
    allowed = route | {s, g}
    dd = {s: 0}
    dq = deque([s])
    while dq:
        cur = dq.popleft()
        if cur == g:
            continue
        for _m, nxt in _nbrs(grid, cur):
            if nxt in allowed and nxt not in dd:
                dd[nxt] = dd[cur] + 1
                dq.append(nxt)
    out["img_valid"] = len(route) == d_sg - 1 and dd.get(g) == d_sg
    out["img_exact"] = route == gold_inner
    out["img_block_acc"] = float(np.mean([(c in route) == (c in gold_inner) for c in frozen])) if frozen else 1.0
    out["n_route_cells"] = len(route)
    out["agree"] = text_inner == route
    out["joint"] = bool(out["text_success"] and out["img_valid"] and out["agree"])
    out["joint_valid"] = bool(out["text_success"] and out["img_valid"])
    out["joint_consistent"] = bool(out["text_shortest"] and out["img_valid"] and out["agree"])
    return out


# ----------------------------------------------------------------------------- data


def random_map(n: int, rng: random.Random) -> tuple[str, ...]:
    """Random map: S and G on two random cells, every other cell a hole with probability p ~ U(0, 0.35)."""
    p = rng.uniform(0.0, 0.35)
    s, g = rng.sample([(r, c) for r in range(n) for c in range(n)], 2)
    return tuple("".join("S" if (r, c) == s else "G" if (r, c) == g else ("H" if rng.random() < p else "F")
                         for c in range(n)) for r in range(n))


def thinkmorph_grid(png_bytes: bytes, sprites: dict[str, np.ndarray]) -> tuple[str, ...]:
    """A ThinkMorph Spatial_Navigation map image (64 px cells) -> grid, by exact tile matching."""
    im = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB")).astype(np.float32)
    n = im.shape[0] // SPRITE
    if im.shape[:2] != (SPRITE * n, SPRITE * n):
        raise ValueError(f"unexpected ThinkMorph image size {im.shape}")
    tiles = {k: v.astype(np.float32) for k, v in sprites.items()}
    rows = []
    for i in range(n):
        row = ""
        for j in range(n):
            crop = im[SPRITE * i:SPRITE * (i + 1), SPRITE * j:SPRITE * (j + 1)]
            devs = {k: float(np.abs(crop - t).mean()) for k, t in tiles.items()}
            k = min(devs, key=devs.get)
            if devs[k] > 1.0:
                raise ValueError(f"cell ({i},{j}) matches no tile (best deviation {devs[k]:.2f})")
            row += k
        rows.append(row)
    return tuple(rows)


def make_record(tokenizer, split: str, idx: int, source: str, grid: Sequence[str], moves: Sequence[str],
                extra: dict | None = None) -> tuple[dict | None, str | None]:
    """One puzzle record, or (None, reason) when a filter rejects it."""
    from juice.representations import TEXT_CANVAS
    from juice.tasks.planning import answer_span

    n = len(grid)
    span = answer_span(tokenizer, question(n), answer_str(moves))
    if span is None:
        return None, "boundary"
    if span["n_content_tokens"] + 1 > TEXT_CANVAS:
        return None, "length"
    if span["answer_tok_end"] - span["answer_tok_start"] > MAX_ANSWER_TOKENS:
        return None, "answer_len"
    status, cells, used = simulate(grid, moves)
    if status != "goal" or used != len(moves):
        return None, "route"
    if len(set(cells)) != len(cells):
        return None, "nonsimple"
    rec = {
        "id": idx, "key": f"{idx:08d}", "split": split, "source": source, "n": n, "grid": list(grid),
        "start": list(find(grid, "S")), "goal": list(find(grid, "G")), "moves": list(moves),
        "answer": answer_str(moves), "question": question(n), **span,
        "img_known_patches": known_patch_flags(grid), "layout_hash": layout_hash(grid),
        "shortest_len": bfs_dist(grid)[find(grid, "G")], "route_cells": [list(c) for c in cells],
    }
    rec.update(extra or {})
    return rec, None


def official_records(maps_dir: str | Path, tokenizer, levels: Sequence[int] = LEVELS) -> list[dict]:
    """The official VSP maze maps as records (100 per level, in map-id order; id = 1000 * level + map id).

    The gold plan is the canonical BFS route; it is used for img/text exactness only,
    never for the success metric.
    """
    maps_dir = Path(maps_dir)
    records = []
    for level in levels:
        for map_id in range(100):
            grid = table_to_grid((maps_dir / f"level{level}" / "table" / f"{map_id}.txt").read_text())
            rec, why = make_record(tokenizer, f"vsp_level{level}", level * 1000 + map_id, "vsp", grid,
                                   bfs_route(grid)[1], {"vsp_level": level, "vsp_map_id": map_id})
            if rec is None:
                raise RuntimeError(f"VSP level {level} map {map_id} dropped ({why})")
            records.append(rec)
    return records
