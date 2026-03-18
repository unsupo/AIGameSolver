import sqlite3
import json
import heapq
from typing import List, Tuple, Dict
from pathlib import Path


class Pathfinder:
    """A* pathfinder that uses the learned SLAM grid (Walkable/Blocked) as nodes."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _get_weighted_map_data(self, map_id: int) -> Dict[Tuple[int, int], float]:
        """
        Queries the long_term_memory.db to retrieve all known tiles for a map
        and their associated impassable_scores.
        """
        grid_costs = {}

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # 1. SLAM data with weighted impassability
                cursor = conn.execute(
                    "SELECT x, y, state, impassable_score FROM explored_locations WHERE map_id = ?",
                    (map_id,),
                )
                for row in cursor:
                    x, y, state, score = row
                    # Prioritize impassable_score, fallback to state
                    actual_score = (
                        score if score is not None else (1.0 if state == 2 else 0.0)
                    )
                    grid_costs[(x, y)] = actual_score

                # 2. Additional walls from replay_buffer (Fallback for hard stalls)
                cursor = conn.execute(
                    "SELECT coords, button FROM replay_buffer WHERE map_id = ? AND stuck = 1",
                    (map_id,),
                )
                for row in cursor:
                    try:
                        coords = json.loads(row[0])
                        btn = row[1].lower()
                        x, y = coords[0], coords[1]
                        # If we were stuck while pressing UP, the tile ABOVE is likely a wall
                        wall_pos = None
                        if btn == "up":
                            wall_pos = (x, y - 1)
                        elif btn == "down":
                            wall_pos = (x, y + 1)
                        elif btn == "left":
                            wall_pos = (x - 1, y)
                        elif btn == "right":
                            wall_pos = (x + 1, y)

                        if wall_pos:
                            grid_costs[wall_pos] = max(grid_costs.get(wall_pos, 0), 0.9)
                    except json.JSONDecodeError, IndexError, TypeError:
                        continue
        except Exception:
            pass

        return grid_costs

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan Distance."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(
        self, map_id: int, start: Tuple[int, int], end: Tuple[int, int]
    ) -> List[str]:
        """Finds the most efficient sequence of buttons using Dijkstra-weighted A* search."""
        if start == end:
            return []

        # 1. Fetch the full map cost data from the SLAM grid
        tile_costs = self._get_weighted_map_data(map_id)

        # Priority Queue for A*: (priority, current_coords, path)
        pq = [(0 + self._heuristic(start, end), start, [])]
        g_score = {start: 0}

        directions = [
            ("up", (0, -1)),
            ("down", (0, 1)),
            ("left", (-1, 0)),
            ("right", (1, 0)),
        ]

        visited = set()

        while pq:
            _, curr, path = heapq.heappop(pq)

            if curr == end:
                return path

            if curr in visited:
                continue
            visited.add(curr)

            if len(path) > 100:
                continue

            curr_x, curr_y = curr
            for btn, (dx, dy) in directions:
                neighbor = (curr_x + dx, curr_y + dy)

                # 2. Dynamic Cost Calculation
                # 0.0 = Walkable, 1.0 = Wall
                # Default 0.05 for UNKNOWN (Low exploration cost)
                tile_penalty = tile_costs.get(neighbor, 0.05)

                if tile_penalty >= 0.9 and neighbor != end:
                    continue  # Solid wall

                # Movement Cost = 1 (step) + (penalty * 10)
                # Weighted cost ensures Dijkstra "wisdom" while A* provides "direction"
                move_cost = 1 + (tile_penalty * 10)
                new_g = g_score[curr] + move_cost

                if neighbor not in g_score or new_g < g_score[neighbor]:
                    g_score[neighbor] = new_g
                    priority = new_g + self._heuristic(neighbor, end)
                    heapq.heappush(pq, (priority, neighbor, path + [btn]))

        return []  # No path found
