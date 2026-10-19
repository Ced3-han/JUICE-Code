"""JMaze: joint maze solving in text and image (paper Sec. 5.4, App. D.1).

Task
----
Input:  a 224 px maze image (walls black, corridors white, start green, goal red)
        and the question "Maze {n}x{n}. Start (r,c). Goal (r,c)." (0-based cells).
        The walls are only in the image.
Output: the same maze with the route painted blue (image) and the move string
        "U U R D" (text). The model makes both outputs in one joint sampling run.
Metric: joint_consistent = the text is a valid shortest route, the image shows a
        valid shortest route, and the two routes are the same route.

Geometry
--------
One maze block is one 14 px encoder patch of the 16 x 16 patch grid. A maze with
n x n cells has 2n + 1 blocks per side (cells at odd indices, wall slots at even
indices), so n <= 7. The maze is centered on a whole patch. Because blocks and
patches line up, the image condition is exact: the background, wall, start and
goal patches are known and only the corridor patches are generated.

Mazes ("braid" topology)
------------------------
A depth-first spanning tree is braided: at each dead end one wall is removed.
The loops give several shortest routes between two cells. A puzzle is kept only
if it has 2 to 16 shortest routes. The training target is one of these routes,
drawn at random per puzzle, for both the image and the text. So the answer has
several modes, and the text and the image must agree on one of them.

Blocks
------
``blocks`` is a tuple of 2n + 1 strings of 2n + 1 characters: "#" = wall,
"." = open. Cell (r, c) is block (2r + 1, 2c + 1). The block between two
neighbor cells is open when the cells are connected. The blue route covers the
route cells and the blocks between them, but not the start and goal blocks.
"""

from __future__ import annotations

import hashlib
import random
from collections import deque
from typing import Sequence

from PIL import Image, ImageDraw

from juice.representations import TEXT_CANVAS
from juice.tasks.planning import answer_span

SIZE = 224
PATCH = 14
GRID = SIZE // PATCH          # 16 patches per side
MAX_N = (GRID - 1) // 2       # 7 cells per side
PARSE_SHRINK = 0.3            # the image readout uses the inner part of each block

COLOR_BG = (255, 255, 255)
COLOR_WALL = (0, 0, 0)
COLOR_CORRIDOR = (255, 255, 255)
COLOR_START = (60, 200, 60)
COLOR_GOAL = (220, 50, 50)
COLOR_PATH = (30, 60, 230)

MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
MOVE_OF_DELTA = {v: k for k, v in MOVES.items()}

# Generator settings of the released data (JMaze-multi).
TRAIN_SIZES = (5, 6, 7)
OOD_SIZES = (3, 4)
MIN_PATH_CELLS = 3            # at least 2 moves
MIN_SHORTEST_ROUTES = 2
MAX_SHORTEST_ROUTES = 16
ROUTE_ENUM_CAP = 64
MAX_ATTEMPTS = 64             # candidates per puzzle slot
MAX_ANSWER_TOKENS = 60


# ----------------------------------------------------------------------------- geometry


def n_blocks(n: int) -> int:
    return 2 * n + 1


def block_offset(n: int) -> int:
    """Pixel offset of block (0, 0): a whole number of patches, maze centered."""
    if n_blocks(n) > GRID:
        raise ValueError(f"n={n} needs {n_blocks(n)} blocks > {GRID} patches; the maximum n is {MAX_N}")
    return PATCH * ((GRID - n_blocks(n)) // 2)


def block_xy(n: int, i: int, j: int) -> tuple[int, int]:
    """Top-left pixel (x, y) of block (row i, column j)."""
    off = block_offset(n)
    return off + PATCH * j, off + PATCH * i


def cell_block(r: int, c: int) -> tuple[int, int]:
    return 2 * r + 1, 2 * c + 1


def between_block(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    """Wall-slot block between two neighbor cells."""
    (r0, c0), (r1, c1) = a, b
    if abs(r0 - r1) + abs(c0 - c1) != 1:
        raise ValueError(f"cells {a} and {b} are not neighbors")
    return r0 + r1 + 1, c0 + c1 + 1


def maze_size(blocks: Sequence[str]) -> int:
    return (len(blocks) - 1) // 2


def is_open(blocks: Sequence[str], i: int, j: int) -> bool:
    return blocks[i][j] == "."


def neighbors(blocks: Sequence[str], r: int, c: int):
    """Cells reachable from cell (r, c) in one move, as (move, (r, c)), in U, D, L, R order."""
    n = maze_size(blocks)
    for move, (dr, dc) in MOVES.items():
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < n and is_open(blocks, *between_block((r, c), (rr, cc))):
            yield move, (rr, cc)


# ----------------------------------------------------------------------------- generation


def gen_maze_blocks(n: int, rng: random.Random) -> tuple[str, ...]:
    """Depth-first spanning tree on n x n cells (one route between any two cells)."""
    nb = n_blocks(n)
    grid = [["#"] * nb for _ in range(nb)]
    visited = [[False] * n for _ in range(n)]
    r0, c0 = rng.randrange(n), rng.randrange(n)
    visited[r0][c0] = True
    br, bc = cell_block(r0, c0)
    grid[br][bc] = "."
    stack = [(r0, c0)]
    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr, c + dc) for dr, dc in MOVES.values()
                if 0 <= r + dr < n and 0 <= c + dc < n and not visited[r + dr][c + dc]]
        if not nbrs:
            stack.pop()
            continue
        rr, cc = rng.choice(nbrs)
        visited[rr][cc] = True
        wr, wc = between_block((r, c), (rr, cc))
        grid[wr][wc] = "."
        br, bc = cell_block(rr, cc)
        grid[br][bc] = "."
        stack.append((rr, cc))
    return tuple("".join(row) for row in grid)


def braid_maze_blocks(blocks: Sequence[str], rng: random.Random) -> tuple[str, ...]:
    """At each dead end of the input maze, remove one random wall to a neighbor cell.

    The dead ends are found once, on the input maze. One rng.random() is drawn per
    dead end (always below the threshold 1.0), which keeps the random stream of the
    paper's data.
    """
    n = maze_size(blocks)
    grid = [list(row) for row in blocks]

    def slots(r: int, c: int):
        for dr, dc in MOVES.values():
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n:
                yield between_block((r, c), (rr, cc))

    dead_ends = [(r, c) for r in range(n) for c in range(n)
                 if sum(1 for wr, wc in slots(r, c) if grid[wr][wc] == ".") <= 1]
    for r, c in dead_ends:
        walled = [(wr, wc) for wr, wc in slots(r, c) if grid[wr][wc] == "#"]
        if rng.random() >= 1.0 or not walled:
            continue
        wr, wc = walled[rng.randrange(len(walled))]
        grid[wr][wc] = "."
    return tuple("".join(row) for row in grid)


def bfs_distances(blocks: Sequence[str], start) -> dict[tuple[int, int], int]:
    start = tuple(start)
    dist = {start: 0}
    dq = deque([start])
    while dq:
        cur = dq.popleft()
        for _move, nxt in neighbors(blocks, *cur):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                dq.append(nxt)
    return dist


def solve_maze(blocks: Sequence[str], start, goal):
    """BFS shortest route as (cells, moves), or None. Ties break by the U, D, L, R order."""
    start, goal = tuple(start), tuple(goal)
    prev: dict = {start: None}
    dq = deque([start])
    while dq:
        cur = dq.popleft()
        if cur == goal:
            break
        for move, nxt in neighbors(blocks, *cur):
            if nxt not in prev:
                prev[nxt] = (cur, move)
                dq.append(nxt)
    if goal not in prev:
        return None
    cells, moves, cur = [goal], [], goal
    while prev[cur] is not None:
        cur, move = prev[cur]
        cells.append(cur)
        moves.append(move)
    return cells[::-1], moves[::-1]


def count_shortest_routes(blocks: Sequence[str], start, goal) -> int:
    """Exact number of shortest routes (dynamic program over the BFS layers)."""
    goal = tuple(goal)
    dist = bfs_distances(blocks, start)
    if goal not in dist:
        return 0
    ways = {tuple(start): 1}
    for cell in sorted(dist, key=lambda c: (dist[c], c)):
        if cell == tuple(start):
            continue
        ways[cell] = sum(ways.get(nxt, 0) for _m, nxt in neighbors(blocks, *cell) if dist.get(nxt, -1) == dist[cell] - 1)
    return ways[goal]


def shortest_routes(blocks: Sequence[str], start, goal, cap: int = ROUTE_ENUM_CAP) -> list[list[tuple[int, int]]]:
    """All shortest routes (lists of cells), in a fixed order, at most `cap`."""
    start, goal = tuple(start), tuple(goal)
    dist = bfs_distances(blocks, start)
    if goal not in dist:
        return []
    routes: list = []
    stack = [(goal, [goal])]
    while stack and len(routes) < cap:
        cur, suffix = stack.pop()
        if cur == start:
            routes.append(suffix)
            continue
        for _move, nxt in neighbors(blocks, *cur):
            if dist.get(nxt, -1) == dist[cur] - 1:
                stack.append((nxt, [nxt] + suffix))
    return routes


def moves_from_cells(cells: Sequence[tuple[int, int]]) -> list[str]:
    return [MOVE_OF_DELTA[(b[0] - a[0], b[1] - a[1])] for a, b in zip(cells, cells[1:])]


def route_blocks(cells: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Open blocks of a route: inner cells and the blocks between cells (no start, no goal)."""
    cells = [tuple(c) for c in cells]
    out = []
    for k, cell in enumerate(cells):
        if 0 < k < len(cells) - 1:
            out.append(cell_block(*cell))
        if k + 1 < len(cells):
            out.append(between_block(cell, cells[k + 1]))
    return out


def maze_key(blocks: Sequence[str], start, goal) -> str:
    """Content hash of (maze, start, goal); test splits exclude every training key."""
    return hashlib.sha1(f"{len(blocks)}:{''.join(blocks)}:{tuple(start)}:{tuple(goal)}".encode("ascii")).hexdigest()


def slot_seed(seed: int, split: str, index: int, attempt: int) -> int:
    """Each candidate has its own seed, so any slot can be made alone and in parallel."""
    digest = hashlib.sha256(f"{seed}|{split}|{index}|{attempt}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def make_candidate(seed: int, split: str, index: int, attempt: int, n: int, tokenizer) -> dict | None:
    """One candidate puzzle, or None when a filter rejects it."""
    rng = random.Random(slot_seed(seed, split, index, attempt))
    blocks = braid_maze_blocks(gen_maze_blocks(n, rng), rng)
    start, goal = rng.sample([(r, c) for r in range(n) for c in range(n)], 2)
    path_cells, _moves = solve_maze(blocks, start, goal)
    if len(path_cells) < MIN_PATH_CELLS:
        return None
    n_routes = count_shortest_routes(blocks, start, goal)
    if not MIN_SHORTEST_ROUTES <= n_routes <= MAX_SHORTEST_ROUTES:
        return None
    routes = shortest_routes(blocks, start, goal)
    route_index = rng.randrange(len(routes))
    path_cells = routes[route_index]
    moves = moves_from_cells(path_cells)
    question, answer = format_question(n, start, goal), format_answer(moves)
    span = answer_span(tokenizer, question, answer)
    if span is None or span["answer_tok_end"] - span["answer_tok_start"] > MAX_ANSWER_TOKENS:
        return None
    if span["n_content_tokens"] + 1 > TEXT_CANVAS:
        return None
    return {
        "id": index, "key": f"{index:08d}", "n": n, "blocks": list(blocks),
        "start": list(start), "goal": list(goal), "path_cells": [list(c) for c in path_cells],
        "moves": moves, "question": question, "answer": answer, **span,
        "maze_hash": maze_key(blocks, start, goal), "topology": "braid",
        "n_shortest_routes": n_routes, "route_index": route_index,
    }


def make_puzzle(seed: int, split: str, index: int, n: int, tokenizer, forbidden: set | frozenset = frozenset()) -> dict | None:
    """First candidate of slot `index` that passes the filters and whose key is not in `forbidden`."""
    for attempt in range(MAX_ATTEMPTS):
        rec = make_candidate(seed, split, index, attempt, n, tokenizer)
        if rec is not None and rec["maze_hash"] not in forbidden:
            return rec
    return None


# ----------------------------------------------------------------------------- render and text


def render_maze(blocks: Sequence[str], start, goal, path_cells: Sequence[tuple[int, int]] | None = None) -> Image.Image:
    """224 px RGB image. path_cells=None gives the input image (no blue route)."""
    n = maze_size(blocks)
    im = Image.new("RGB", (SIZE, SIZE), COLOR_BG)
    draw = ImageDraw.Draw(im)

    def fill(i: int, j: int, color) -> None:
        x, y = block_xy(n, i, j)
        draw.rectangle([x, y, x + PATCH - 1, y + PATCH - 1], fill=color)

    for i in range(n_blocks(n)):
        for j in range(n_blocks(n)):
            fill(i, j, COLOR_CORRIDOR if is_open(blocks, i, j) else COLOR_WALL)
    if path_cells:
        for i, j in route_blocks(path_cells):
            fill(i, j, COLOR_PATH)
    fill(*cell_block(*start), COLOR_START)
    fill(*cell_block(*goal), COLOR_GOAL)
    return im


def known_patch_flags(blocks: Sequence[str], start, goal) -> list[int]:
    """256 ints in patch order (row-major): 1 = known (background, wall, start, goal), 0 = generated."""
    n = maze_size(blocks)
    off = block_offset(n) // PATCH
    flags = [1] * (GRID * GRID)
    markers = {cell_block(*start), cell_block(*goal)}
    for i in range(n_blocks(n)):
        for j in range(n_blocks(n)):
            if blocks[i][j] == "." and (i, j) not in markers:
                flags[(off + i) * GRID + (off + j)] = 0
    return flags


def format_question(n: int, start, goal) -> str:
    return f"Maze {n}x{n}. Start ({start[0]},{start[1]}). Goal ({goal[0]},{goal[1]})."


def format_answer(moves: Sequence[str]) -> str:
    return " ".join(moves)


def parse_answer(text: str) -> list[str] | None:
    """Answer text -> move list, or None when a token is not U, D, L or R."""
    parts = str(text).split()
    if not parts or any(p not in MOVES for p in parts):
        return None
    return parts


# ----------------------------------------------------------------------------- scoring


def replay_moves(blocks: Sequence[str], start, moves: Sequence[str]) -> dict:
    """Walk the moves on the true maze; stop at the first move off the grid or into a wall."""
    n = maze_size(blocks)
    cur = tuple(start)
    cells = [cur]
    for mv in moves:
        dr, dc = MOVES[mv]
        nxt = (cur[0] + dr, cur[1] + dc)
        if not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or not is_open(blocks, *between_block(cur, nxt)):
            return {"legal": False, "end": cur, "cells": cells}
        cur = nxt
        cells.append(cur)
    return {"legal": True, "end": cur, "cells": cells}


def score_text(blocks: Sequence[str], start, goal, moves: Sequence[str] | None) -> dict:
    """valid = a legal walk to the goal with the shortest length; exact = the BFS route."""
    gold_moves = solve_maze(blocks, start, goal)[1]
    if moves is None:
        return {"parse_fail": True, "valid": False, "exact": False, "cells": None}
    walk = replay_moves(blocks, start, moves)
    valid = walk["legal"] and walk["end"] == tuple(goal) and len(moves) == len(gold_moves)
    return {"parse_fail": False, "valid": bool(valid), "exact": list(moves) == gold_moves,
            "cells": walk["cells"] if valid else None}


def classify_block(rgb) -> str:
    """Mean RGB in [0, 1] of a block interior -> path / start / goal / wall / corridor."""
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    if b > r + 0.15 and b > g + 0.15:
        return "path"
    if g > r + 0.15 and g > b + 0.15:
        return "start"
    if r > g + 0.15 and r > b + 0.15:
        return "goal"
    if (r + g + b) / 3.0 < 0.4:
        return "wall"
    return "corridor"


def parse_blocks_from_image(arr, n: int) -> list[list[str]]:
    """Image [H, W, 3] float in [0, 1] -> (2n+1) x (2n+1) block labels."""
    d = PARSE_SHRINK * PATCH
    labels = []
    for i in range(n_blocks(n)):
        row = []
        for j in range(n_blocks(n)):
            x, y = block_xy(n, i, j)
            x0, y0, x1, y1 = int(x + d), int(y + d), int(x + PATCH - d), int(y + PATCH - d)
            row.append(classify_block(arr[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)].reshape(-1, 3).mean(0)))
        labels.append(row)
    return labels


def path_is_valid(blocks: Sequence[str], start, goal, blue: set) -> bool:
    """Do the blue blocks form one shortest route from start to goal, with no extra blue block?"""
    start, goal = tuple(start), tuple(goal)
    if start == goal:
        return not blue
    dist = bfs_distances(blocks, start)
    if goal not in dist:
        return False
    used: set = set()
    cur, steps, seen = start, 0, {start}
    while True:
        if cur == goal:
            return used == blue and steps == dist[goal]
        options = []
        for _move, nxt in neighbors(blocks, *cur):
            wb, cb = between_block(cur, nxt), cell_block(*nxt)
            if nxt in seen or wb not in blue or (nxt != goal and cb not in blue):
                continue
            options.append((nxt, wb, cb))
        if len(options) != 1:
            return False
        nxt, wb, cb = options[0]
        used.add(wb)
        if nxt != goal:
            used.add(cb)
        seen.add(nxt)
        cur, steps = nxt, steps + 1


def score_puzzle(blocks: Sequence[str], start, goal, labels, moves: Sequence[str] | None) -> dict:
    """Scores of one (image labels, text moves) pair. The headline metric is joint_consistent."""
    blocks, start, goal = tuple(blocks), tuple(start), tuple(goal)
    nb = len(blocks)
    gold_cells = solve_maze(blocks, start, goal)[0]
    gold = [["wall" if blocks[i][j] == "#" else "corridor" for j in range(nb)] for i in range(nb)]
    for i, j in route_blocks(gold_cells):
        gold[i][j] = "path"
    gold[cell_block(*start)[0]][cell_block(*start)[1]] = "start"
    gold[cell_block(*goal)[0]][cell_block(*goal)[1]] = "goal"

    txt = score_text(blocks, start, goal, moves)
    blue = {(i, j) for i, row in enumerate(labels) for j, lab in enumerate(row) if lab == "path"}
    img_valid = path_is_valid(blocks, start, goal, blue)
    text_blue = set(route_blocks(txt["cells"])) if txt["cells"] is not None else None
    return {
        "n": maze_size(blocks),
        "text_parse_fail": txt["parse_fail"],
        "text_valid": txt["valid"],
        "text_exact": txt["exact"],
        "img_valid": bool(img_valid),
        "img_exact": blue == set(route_blocks(gold_cells)),
        "joint_valid": bool(txt["valid"] and img_valid),
        "joint": bool(txt["exact"] and blue == set(route_blocks(gold_cells))),
        "agree": text_blue is not None and text_blue == blue,
        "joint_consistent": bool(txt["valid"] and img_valid and text_blue == blue),
        "img_block_acc": sum(labels[i][j] == gold[i][j] for i in range(nb) for j in range(nb)) / float(nb * nb),
        "img_wall_acc": sum((labels[i][j] == "wall") == (blocks[i][j] == "#") for i in range(nb) for j in range(nb)) / float(nb * nb),
        "img_blue": sorted([i, j] for i, j in blue),
        "text_blue": sorted([i, j] for i, j in text_blue) if text_blue is not None else None,
    }
