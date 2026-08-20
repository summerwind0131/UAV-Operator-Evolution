import argparse
import heapq
import json
import math
import random


def read_problem(path: str) -> dict:
    with open(path, 'r') as f:
        data = json.load(f)

    env = data['environment']
    width = float(env['width'])
    height = float(env['height'])
    start = tuple(float(x) for x in env['start'])
    goal = tuple(float(x) for x in env['goal'])

    obstacles = []
    for o in env.get('obstacles', []):
        kind = o.get('kind', '')
        if kind == 'circle':
            obstacles.append({
                'kind': 'circle',
                'center': tuple(float(c) for c in o['center']),
                'radius': float(o['radius'])
            })
        elif kind == 'rectangle':
            obstacles.append({
                'kind': 'rectangle',
                'min_x': float(o['min_x']),
                'min_y': float(o['min_y']),
                'max_x': float(o['max_x']),
                'max_y': float(o['max_y'])
            })
        else:
            raise ValueError(f"Unknown obstacle kind: {kind}")

    risk_zones = [
        {
            'min_x': float(rz['min_x']),
            'max_x': float(rz['max_x']),
            'min_y': float(rz['min_y']),
            'max_y': float(rz['max_y']),
            'weight': float(rz['weight']),
            'name': rz.get('name', 'unnamed')
        }
        for rz in env.get('risk_zones', [])
    ]
    safety_distance = float(env.get('safety_distance', 0.0))
    seed = int(env.get('seed', data.get('seed', 0)))
    objective_weights = {k: float(v) for k, v in data['objective_weights'].items()}
    grid_resolution = float(data.get('grid_resolution', 1.0))
    max_waypoints = int(data.get('max_waypoints', 100))

    return {
        'width': width,
        'height': height,
        'start': start,
        'goal': goal,
        'obstacles': obstacles,
        'risk_zones': risk_zones,
        'safety_distance': safety_distance,
        'seed': seed,
        'objective_weights': objective_weights,
        'grid_resolution': grid_resolution,
        'max_waypoints': max_waypoints,
        '_metrics': {}
    }


def segment_collision_free(start, end, problem) -> bool:
    """Check if segment is collision-free against hard obstacles.
    Risk zones are ignored. Counts one collision check."""
    metrics = problem.get('_metrics')
    if isinstance(metrics, dict):
        metrics['collision_checks'] = metrics.get('collision_checks', 0) + 1

    safety = problem.get('safety_distance', 0.0)
    obstacles = problem.get('obstacles', [])
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    len_sq = dx * dx + dy * dy

    for obs in obstacles:
        kind = obs.get('kind')
        if kind == 'circle':
            cx, cy = obs['center']
            r = obs['radius'] + safety
            if len_sq == 0:
                dist_sq = (start[0] - cx) ** 2 + (start[1] - cy) ** 2
                if dist_sq < r * r:
                    return False
                continue
            t = ((cx - start[0]) * dx + (cy - start[1]) * dy) / len_sq
            t = max(0.0, min(1.0, t))
            closest_x = start[0] + t * dx
            closest_y = start[1] + t * dy
            dist_sq = (closest_x - cx) ** 2 + (closest_y - cy) ** 2
            if dist_sq < r * r:
                return False
        elif kind == 'rectangle':
            min_x = obs['min_x'] - safety
            max_x = obs['max_x'] + safety
            min_y = obs['min_y'] - safety
            max_y = obs['max_y'] + safety
            p = [-dx, dx, -dy, dy]
            q = [start[0] - min_x, max_x - start[0], start[1] - min_y, max_y - start[1]]
            u1 = 0.0
            u2 = 1.0
            intersects = True
            for i in range(4):
                if p[i] == 0:
                    if q[i] < 0:
                        intersects = False
                        break
                else:
                    r = q[i] / p[i]
                    if p[i] < 0:
                        if r > u2:
                            intersects = False
                            break
                        if r > u1:
                            u1 = r
                    else:
                        if r < u1:
                            intersects = False
                            break
                        if r < u2:
                            u2 = r
            if intersects and u1 <= u2:
                return False
    return True


def _segment_risk_exposure(start, end, risk_zones) -> float:
    """Continuous weighted length inside rectangular risk zones.
    Zero-length segment returns 0."""
    total = 0.0
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    seg_len = math.hypot(dx, dy)
    if seg_len == 0:
        return 0.0
    for zone in risk_zones:
        min_x, max_x = zone['min_x'], zone['max_x']
        min_y, max_y = zone['min_y'], zone['max_y']
        t_enter = 0.0
        t_exit = 1.0
        inside = True
        if dx == 0:
            if start[0] < min_x or start[0] > max_x:
                inside = False
        else:
            t1 = (min_x - start[0]) / dx
            t2 = (max_x - start[0]) / dx
            if t1 > t2:
                t1, t2 = t2, t1
            t_enter = max(t_enter, t1)
            t_exit = min(t_exit, t2)
        if dy == 0:
            if start[1] < min_y or start[1] > max_y:
                inside = False
        else:
            t1 = (min_y - start[1]) / dy
            t2 = (max_y - start[1]) / dy
            if t1 > t2:
                t1, t2 = t2, t1
            t_enter = max(t_enter, t1)
            t_exit = min(t_exit, t2)
        if inside and t_enter < t_exit:
            total += zone['weight'] * (t_exit - t_enter) * seg_len
    return total


def path_cost(path, problem) -> float:
    """Evaluate objective and increment objective_evaluations exactly once.
    Uses counted segment_collision_free for collision tests."""
    metrics = problem.get('_metrics')
    if isinstance(metrics, dict):
        metrics['objective_evaluations'] = metrics.get('objective_evaluations', 0) + 1

    w = problem.get('objective_weights', {})
    w_length = w.get('length', 0.0)
    w_collision = w.get('collision', 0.0)
    w_smoothness = w.get('smoothness', 0.0)
    w_risk = w.get('risk', 0.0)
    w_waypoint = w.get('waypoint', 0.0)

    n = len(path)
    total = w_waypoint * max(0, n - 2)
    if n == 0:
        return total

    length_term = 0.0
    collision_term = 0.0
    smoothness_term = 0.0
    risk_term = 0.0

    for i in range(n - 1):
        seg_len = math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        length_term += seg_len
        if not segment_collision_free(path[i], path[i+1], problem):
            collision_term += 1.0
        risk_term += _segment_risk_exposure(path[i], path[i+1], problem.get('risk_zones', []))

    if n >= 3:
        for i in range(1, n - 1):
            vx0 = path[i][0] - path[i-1][0]
            vy0 = path[i][1] - path[i-1][1]
            vx1 = path[i+1][0] - path[i][0]
            vy1 = path[i+1][1] - path[i][1]
            len0 = math.hypot(vx0, vy0)
            len1 = math.hypot(vx1, vy1)
            if len0 == 0 or len1 == 0:
                smoothness_term += 1.0  # degenerate turn penalty
            else:
                dot = vx0 * vx1 + vy0 * vy1
                cos_angle = max(-1.0, min(1.0, dot / (len0 * len1)))
                angle = math.acos(cos_angle)
                smoothness_term += (angle / math.pi) ** 2

    return (w_length * length_term +
            w_collision * collision_term +
            w_smoothness * smoothness_term +
            w_risk * risk_term +
            total)


def initial_path(problem) -> list:
    width = problem['width']
    height = problem['height']
    start = problem['start']
    goal = problem['goal']
    res = problem['grid_resolution']

    nx = int(math.floor(width / res)) + 1
    ny = int(math.floor(height / res)) + 1
    if nx < 1 or ny < 1:
        raise RuntimeError('grid too small')

    def node_to_point(node):
        i, j = node
        return [i * res, j * res]

    def point_to_node(point):
        x, y = point
        return int(round(x / res)), int(round(y / res))

    def is_inside_map(point):
        x, y = point
        return 0.0 <= x <= width and 0.0 <= y <= height

    def is_valid_node(node):
        i, j = node
        return 0 <= i <= nx - 1 and 0 <= j <= ny - 1

    if not is_inside_map(start) or not is_inside_map(goal):
        raise RuntimeError('start or goal outside map')

    start_node = point_to_node(start)
    goal_node = point_to_node(goal)

    if not is_valid_node(start_node) or not is_valid_node(goal_node):
        raise RuntimeError('start or goal node outside grid')

    if start_node == goal_node:
        return [start, goal]

    if segment_collision_free(start, goal, problem):
        return [start, goal]

    counter = 0
    open_set = []
    heapq.heappush(open_set, (0.0, counter, start_node))
    came_from = {}
    g_score = {start_node: 0.0}
    closed_set = set()

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1),
                 (1, 1), (1, -1), (-1, 1), (-1, -1)]
    metrics = problem.get('_metrics')

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current in closed_set:
            continue
        closed_set.add(current)
        if isinstance(metrics, dict):
            metrics['node_expansions'] = metrics.get('node_expansions', 0) + 1
        if current == goal_node:
            break

        current_point = node_to_point(current)
        for di, dj in neighbors:
            neighbor = (current[0] + di, current[1] + dj)
            if not is_valid_node(neighbor) or neighbor in closed_set:
                continue
            neighbor_point = node_to_point(neighbor)
            if not is_inside_map(neighbor_point):
                continue
            if not segment_collision_free(current_point, neighbor_point, problem):
                continue
            step_cost = math.hypot(di * res, dj * res)
            tentative_g = g_score[current] + step_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = math.hypot(neighbor_point[0] - goal[0], neighbor_point[1] - goal[1])
                tie = 1e-6 * (neighbor[0] * 0.171 + neighbor[1] * 0.319)
                counter += 1
                heapq.heappush(open_set, (tentative_g + h + tie, counter, neighbor))

    if goal_node not in came_from:
        raise RuntimeError('no feasible path found by grid search')

    path_nodes = [goal_node]
    current = goal_node
    while current != start_node:
        current = came_from[current]
        path_nodes.append(current)
    path_nodes.reverse()
    path_points = [node_to_point(node) for node in path_nodes]
    path_points[0] = start
    path_points[-1] = goal

    max_waypoints = problem.get('max_waypoints')
    if max_waypoints is not None and len(path_points) > max_waypoints:
        # Simplify while preserving feasibility and endpoint order.
        simplified = [path_points[0]]
        i = 1
        while i < len(path_points):
            last = simplified[-1]
            j = len(path_points) - 1
            while j > i and not segment_collision_free(last, path_points[j], problem):
                j -= 1
            if j > i:
                simplified.append(path_points[j])
                i = j + 1
            else:
                simplified.append(path_points[i])
                i += 1
        # Ensure no consecutive duplicates, especially before appending goal if needed.
        deduped = [simplified[0]]
        for p in simplified[1:]:
            if p != deduped[-1]:
                deduped.append(p)
        simplified = deduped
        if simplified[-1] != goal:
            simplified.append(goal)
        path_points = simplified

    # Remove consecutive duplicates that may have been introduced.
    deduped_path = [path_points[0]]
    for p in path_points[1:]:
        if p != deduped_path[-1]:
            deduped_path.append(p)
    path_points = deduped_path

    while max_waypoints is not None and len(path_points) > max_waypoints:
        new_path = [path_points[0]]
        for i in range(1, len(path_points) - 1):
            if segment_collision_free(new_path[-1], path_points[i+1], problem):
                # Skip this point
                pass
            else:
                new_path.append(path_points[i])
        new_path.append(path_points[-1])
        if len(new_path) >= len(path_points):
            raise RuntimeError('cannot satisfy waypoint limit')
        path_points = new_path

    for i in range(len(path_points) - 1):
        if not segment_collision_free(path_points[i], path_points[i+1], problem):
            raise RuntimeError('post-processed path has collision')
    for p in path_points:
        if not is_inside_map(p):
            raise RuntimeError('post-processed path out of bounds')

    return path_points


def destroy(path, rng):
    if len(path) <= 2:
        return path
    idx = rng.randint(1, len(path) - 2)
    return path[:idx] + path[idx+1:]


def repair(original, candidate, problem, rng):
    """Repair infeasible candidate without objective evaluation."""
    start = problem['start']
    goal = problem['goal']
    max_waypoints = problem.get('max_waypoints')

    if not (len(candidate) >= 2 and candidate[0] == start and candidate[-1] == goal):
        return original
    if len(candidate) > max_waypoints:
        return original

    def is_feasible(path):
        if len(path) > max_waypoints:
            return False
        width, height = problem['width'], problem['height']
        for pt in path:
            if not (math.isfinite(pt[0]) and math.isfinite(pt[1])):
                return False
            if not (0 <= pt[0] <= width and 0 <= pt[1] <= height):
                return False
        for i in range(len(path) - 1):
            if not segment_collision_free(path[i], path[i+1], problem):
                return False
        return True

    if is_feasible(candidate):
        return candidate if len(candidate) <= len(original) else original

    repaired = list(candidate)
    width, height = problem['width'], problem['height']

    def point_feasible(pt):
        if not (math.isfinite(pt[0]) and math.isfinite(pt[1])):
            return False
        if not (0 <= pt[0] <= width and 0 <= pt[1] <= height):
            return False
        return True

    # Attempt to move infeasible intermediate points.
    for i in range(1, len(repaired)-1):
        if not point_feasible(repaired[i]):
            mid = [(repaired[i-1][0] + repaired[i+1][0]) / 2,
                   (repaired[i-1][1] + repaired[i+1][1]) / 2]
            if point_feasible(mid):
                repaired[i] = mid
            else:
                for _ in range(10):
                    dx = rng.uniform(-2.0, 2.0)
                    dy = rng.uniform(-2.0, 2.0)
                    new_pt = [repaired[i][0] + dx, repaired[i][1] + dy]
                    if point_feasible(new_pt):
                        repaired[i] = new_pt
                        break

    # Remove consecutive duplicates that might cause zero-length segments.
    deduped = [repaired[0]]
    for p in repaired[1:]:
        if p != deduped[-1]:
            deduped.append(p)
    repaired = deduped

    return repaired if is_feasible(repaired) else original


def validate_path(path, problem) -> bool:
    if not path or len(path) < 2:
        return False
    max_waypoints = problem.get('max_waypoints')
    if max_waypoints is not None and len(path) > max_waypoints:
        return False
    start = problem['start']
    goal = problem['goal']
    if tuple(path[0]) != tuple(start) or tuple(path[-1]) != tuple(goal):
        return False
    width = problem['width']
    height = problem['height']

    for pt in path:
        if len(pt) != 2:
            return False
        x, y = pt
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        if x < 0 or x > width or y < 0 or y > height:
            return False

    for i in range(len(path) - 1):
        if not segment_collision_free(path[i], path[i+1], problem):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--iteration', type=int, default=100)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max-evaluations', type=int, default=1000)
    args = parser.parse_args()

    problem = read_problem(args.path)
    problem['_metrics'] = {'objective_evaluations': 0, 'collision_checks': 0, 'node_expansions': 0}

    seed = args.seed if args.seed is not None else problem['seed']
    rng = random.Random(seed)

    path = initial_path(problem)
    if not validate_path(path, problem):
        raise RuntimeError('initial_path infeasible')
    best_path = path.copy()

    best_cost = path_cost(path, problem)
    initial_cost = best_cost
    iterations = 0

    for _ in range(args.iteration):
        if problem['_metrics']['objective_evaluations'] >= args.max_evaluations:
            break
        candidate = destroy(path, rng)
        repaired = repair(path, candidate, problem, rng)
        if not validate_path(repaired, problem):
            iterations += 1
            continue
        if problem['_metrics']['objective_evaluations'] >= args.max_evaluations:
            break
        repaired_cost = path_cost(repaired, problem)
        if repaired_cost < best_cost:
            path = repaired
            best_cost = repaired_cost
            best_path = repaired.copy()
        iterations += 1

    if not validate_path(best_path, problem):
        raise RuntimeError('best path became infeasible')

    result = {
        'status': 'success',
        'path': best_path,
        'initial_cost': initial_cost,
        'best_cost': best_cost,
        'iterations': iterations,
        'seed': seed,
        'objective_evaluations': problem['_metrics']['objective_evaluations'],
        'collision_checks': problem['_metrics']['collision_checks'],
        'node_expansions': problem['_metrics']['node_expansions']
    }
    with open(args.output, 'w') as f:
        json.dump(result, f)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
