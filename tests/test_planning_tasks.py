"""CPU unit tests for the planning tasks: generation, rendering, parsing and scoring.

    python -m unittest tests.test_planning_tasks -v
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from juice.tasks import maze, vsp
from juice.tasks.planning import answer_span, images_to_unit


class WordTokenizer:
    """Stand-in for the T5 tokenizer: one id per space-separated word."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [sum(map(ord, w)) % 1000 + 2 for w in text.split()]}


def as_float(image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


class MazeTest(unittest.TestCase):
    # First puzzle of the released test_id split (seed 0).
    BLOCKS = ("###########", "#.........#", "#.#.#.###.#", "#...#.....#", "#.#.#####.#", "#.#.#...#.#",
              "#.#.#.#.#.#", "#.#...#...#", "#.#.###.#.#", "#.........#", "###########")

    def test_generator_is_deterministic_and_matches_the_released_split(self):
        rec = maze.make_puzzle(0, "test_id", 0, 5, WordTokenizer())
        self.assertEqual(tuple(rec["blocks"]), self.BLOCKS)
        self.assertEqual((rec["start"], rec["goal"]), ([3, 0], [1, 2]))
        self.assertEqual(rec["answer"], "U U U R R D")
        self.assertEqual(rec["question"], "Maze 5x5. Start (3,0). Goal (1,2).")
        self.assertEqual(rec["n_shortest_routes"], maze.count_shortest_routes(self.BLOCKS, (3, 0), (1, 2)))
        self.assertGreaterEqual(rec["n_shortest_routes"], 2)

    def test_render_is_patch_aligned(self):
        rec = maze.make_puzzle(0, "test_id", 0, 5, WordTokenizer())
        cells = [tuple(c) for c in rec["path_cells"]]
        arr = np.asarray(maze.render_maze(self.BLOCKS, (3, 0), (1, 2), cells))
        off = maze.block_offset(5)
        self.assertEqual(off % maze.PATCH, 0)
        x, y = maze.block_xy(5, *maze.cell_block(3, 0))
        self.assertTrue((arr[y:y + 14, x:x + 14] == maze.COLOR_START).all())
        x, y = maze.block_xy(5, 0, 0)
        self.assertTrue((arr[y:y + 14, x:x + 14] == maze.COLOR_WALL).all())
        flags = maze.known_patch_flags(self.BLOCKS, (3, 0), (1, 2))
        n_open = sum(row.count(".") for row in self.BLOCKS)
        self.assertEqual(flags.count(0), n_open - 2)

    def test_clean_render_scores_as_solved(self):
        start, goal = (3, 0), (1, 2)
        for route in maze.shortest_routes(self.BLOCKS, start, goal):
            labels = maze.parse_blocks_from_image(as_float(maze.render_maze(self.BLOCKS, start, goal, route)), 5)
            moves = maze.parse_answer(maze.format_answer(maze.moves_from_cells(route)))
            s = maze.score_puzzle(self.BLOCKS, start, goal, labels, moves)
            self.assertTrue(s["joint_consistent"] and s["img_valid"] and s["text_valid"])
            self.assertEqual(s["img_wall_acc"], 1.0)

    def test_scores_catch_errors(self):
        start, goal = (3, 0), (1, 2)
        routes = maze.shortest_routes(self.BLOCKS, start, goal)
        labels = maze.parse_blocks_from_image(as_float(maze.render_maze(self.BLOCKS, start, goal, routes[0])), 5)
        other = maze.moves_from_cells(routes[1])
        s = maze.score_puzzle(self.BLOCKS, start, goal, labels, other)
        self.assertTrue(s["text_valid"] and s["img_valid"])
        self.assertFalse(s["agree"] or s["joint_consistent"])          # valid, but two different routes
        s = maze.score_puzzle(self.BLOCKS, start, goal, labels, ["U", "U"])
        self.assertFalse(s["text_valid"])
        input_labels = maze.parse_blocks_from_image(as_float(maze.render_maze(self.BLOCKS, start, goal)), 5)
        self.assertFalse(maze.score_puzzle(self.BLOCKS, start, goal, input_labels, None)["img_valid"])
        self.assertIsNone(maze.parse_answer("U X D"))
        self.assertIsNone(maze.parse_answer(""))
        self.assertEqual(maze.parse_answer(" U  D "), ["U", "D"])


class VSPTest(unittest.TestCase):
    GRID = ("SFF", "FHF", "FFG")
    TILES = {k: np.full((vsp.CELL, vsp.CELL, 3), v, dtype=np.uint8)
             for k, v in {"S": (0, 200, 0), "F": (200, 220, 255), "H": (20, 20, 80), "G": (240, 200, 0)}.items()}

    def test_table_to_grid(self):
        table = "| | Col 1 | Col 2 | Col 3 |\n| Row 1 | _ | * | _ |\n| Row 2 | _ | # | _ |\n| Row 3 | _ | @ | _ |"
        self.assertEqual(vsp.table_to_grid(table), ("FGF", "FHF", "FSF"))

    def test_simulate_follows_frozenlake(self):
        self.assertEqual(vsp.simulate(self.GRID, ["R", "R", "D", "D"])[0], "goal")
        self.assertEqual(vsp.simulate(self.GRID, ["D", "R"])[0], "hole")
        self.assertEqual(vsp.simulate(self.GRID, ["U", "L"])[2], 2)        # off the map: stay in place
        self.assertEqual(vsp.simulate(self.GRID, ["R", "R", "D", "D", "X"])[0], "goal")  # moves after the goal are ignored
        self.assertEqual(vsp.simulate(self.GRID, ["X", "R"])[0], "invalid")
        self.assertEqual(vsp.parse_answer(' "R", R.,D\n'), ["R", "R", "D"])

    def test_bfs_route_and_flags(self):
        cells, moves = vsp.bfs_route(self.GRID)
        self.assertEqual(len(moves), 4)
        self.assertEqual(vsp.answer_str(moves), ",".join(moves))
        flags = vsp.known_patch_flags(self.GRID)
        self.assertEqual(flags.count(0), 4 * sum(row.count("F") for row in self.GRID))

    def test_render_readout_and_score(self):
        cells, moves = vsp.bfs_route(self.GRID)
        solved = vsp.render(self.GRID, self.TILES, cells)
        route = vsp.route_cells_from_image(as_float(solved), self.GRID)
        self.assertEqual(route, set(cells[1:-1]))
        self.assertEqual(vsp.route_cells_from_image(np.asarray(solved), self.GRID), route)   # uint8 input
        self.assertEqual(vsp.route_cells_from_image(as_float(vsp.render(self.GRID, self.TILES)), self.GRID), set())
        s = vsp.score(self.GRID, moves, route, vsp.answer_str(moves))
        self.assertTrue(s["text_success"] and s["img_valid"] and s["joint_consistent"])
        s = vsp.score(self.GRID, moves, route, "D,R")
        self.assertFalse(s["text_success"] or s["joint"])


class PlanningTest(unittest.TestCase):
    def test_answer_span(self):
        span = answer_span(WordTokenizer(), "Maze 5x5. Start (3,0). Goal (1,2).", "U U R")
        self.assertEqual(span["caption"], "Question: Maze 5x5. Start (3,0). Goal (1,2). Answer: U U R")
        self.assertEqual(span["answer_tok_end"] - span["answer_tok_start"], 3)
        self.assertIsNone(answer_span(WordTokenizer(), "q", ""))

    def test_images_to_unit(self):
        x = images_to_unit([np.array([[[0, 255, 51]]], dtype=np.uint8)])
        self.assertEqual(tuple(x.shape), (1, 3, 1, 1))
        self.assertTrue(torch.allclose(x.flatten(), torch.tensor([-1.0, 1.0, 51 / 255 * 2 - 1])))


if __name__ == "__main__":
    unittest.main()
