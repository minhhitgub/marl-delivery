import numpy as np
import random
from collections import defaultdict

def run_bfs(map, start, goal):
    n_rows = len(map)
    n_cols = len(map[0])

    queue = []
    visited = set()
    queue.append((goal, []))
    visited.add(goal)
    d = {}
    d[goal] = 0

    while queue:
        current, path = queue.pop(0)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (current[0] + dx, current[1] + dy)
            if next_pos[0] < 0 or next_pos[0] >= n_rows or next_pos[1] < 0 or next_pos[1] >= n_cols:
                continue
            if next_pos not in visited and map[next_pos[0]][next_pos[1]] == 0:
                visited.add(next_pos)
                d[next_pos] = d[current] + 1
                queue.append((next_pos, path + [next_pos]))

    if start not in d:
        return 'S', 100000

    t = 0
    actions = ['U', 'D', 'L', 'R']
    current = start
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        next_pos = (current[0] + dx, current[1] + dy)
        if next_pos in d and d[next_pos] == d[current] - 1:
            return actions[t], d[next_pos]
        t += 1
    return 'S', d[start]

class ACOAgents:

    def __init__(self):
        self.n_robots = 0
        self.robots = []
        self.robots_target = []
        self.packages = []
        self.packages_free = []
        self.map = []
        self.is_init = False
        self.pheromone = defaultdict(lambda: 1.0)
        self.evaporation = 0.1
        self.alpha = 1.0
        self.beta = 2.0

    def init_agents(self, state):
        self.n_robots = len(state['robots'])
        self.map = state['map']
        self.robots = [(robot[0]-1, robot[1]-1, 0) for robot in state['robots']]
        self.robots_target = ['free'] * self.n_robots
        self.packages = [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5]) for p in state['packages']]
        self.packages_free = [True] * len(self.packages)

    def update_inner_state(self, state):
        for i in range(len(state['robots'])):
            prev = self.robots[i]
            robot = state['robots'][i]
            self.robots[i] = (robot[0]-1, robot[1]-1, robot[2])
            if prev[2] != 0 and self.robots[i][2] == 0:
                self.robots_target[i] = 'free'
            elif self.robots[i][2] != 0:
                self.robots_target[i] = self.robots[i][2]
        self.packages += [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5]) for p in state['packages']]
        self.packages_free += [True] * len(state['packages'])

    def compute_score(self, robot, package_idx):
        pkg = self.packages[package_idx]
        d = abs(pkg[1]-robot[0]) + abs(pkg[2]-robot[1])
        tau = self.pheromone[(robot[0], robot[1], pkg[1], pkg[2])]
        return (tau ** self.alpha) * ((1.0 / (d + 1e-6)) ** self.beta)

    def update_move_to_target(self, robot_id, target_package_id, phase='start'):
        pkg = self.packages[target_package_id]
        target_p = (pkg[1], pkg[2]) if phase == 'start' else (pkg[3], pkg[4])
        move, distance = run_bfs(self.map, (self.robots[robot_id][0], self.robots[robot_id][1]), target_p)
        pkg_act = 0
        if distance == 0:
            pkg_act = 1 if phase == 'start' else 2
        return move, str(pkg_act)

    def evaporate_pheromone(self):
        for key in list(self.pheromone.keys()):
            self.pheromone[key] *= (1.0 - self.evaporation)
            if self.pheromone[key] < 0.1:
                del self.pheromone[key]

    def get_actions(self, state):
        if not self.is_init:
            self.is_init = True
            self.init_agents(state)
        else:
            self.update_inner_state(state)

        actions = []
        self.evaporate_pheromone()

        for i in range(self.n_robots):
            if self.robots_target[i] != 'free':
                target_id = self.robots_target[i] - 1
                phase = 'target' if self.robots[i][2] != 0 else 'start'
                move, action = self.update_move_to_target(i, target_id, phase)
                actions.append((move, action))

                # Reinforce pheromone
                pkg = self.packages[target_id]
                start = (self.robots[i][0], self.robots[i][1])
                end = (pkg[3], pkg[4]) if phase == 'target' else (pkg[1], pkg[2])
                self.pheromone[(start[0], start[1], end[0], end[1])] += 1.0
            else:
                # Select package based on pheromone score (probabilistic)
                scores = []
                candidates = []
                for j in range(len(self.packages)):
                    if self.packages_free[j]:
                        score = self.compute_score(self.robots[i], j)
                        scores.append(score)
                        candidates.append(j)
                if not candidates:
                    actions.append(('S', '0'))
                    continue
                probs = np.array(scores) / sum(scores)
                selected_idx = np.random.choice(candidates, p=probs)
                self.packages_free[selected_idx] = False
                self.robots_target[i] = self.packages[selected_idx][0]
                move, action = self.update_move_to_target(i, selected_idx)
                actions.append((move, action))
        print("N robots = ", len(self.robots))
        print("Actions = ", actions)
        print(self.robots_target)
        return actions
