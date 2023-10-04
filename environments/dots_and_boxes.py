from __future__ import annotations

import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

from .grid import Grid


def env(**kwargs):
    env = raw_env(**kwargs)
    # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "dots_and_boxes",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(self, render_mode: str | None = None, screen_scaling: int = 9, rows: int = 2, columns: int = 2, num_agents: int = 2):
        EzPickle.__init__(self, render_mode, screen_scaling)
        super().__init__()
        self.screen = None
        self.render_mode = render_mode
        self.screen_scaling = screen_scaling

        self._rows = rows if rows <= 10 else 10  # The number of rows in the grid
        # The number of columns in the grid
        self._columns = columns if columns <= 10 else 10
        self._num_agents = num_agents if num_agents >= 2 and num_agents <= 6 else (
            2 if num_agents < 2 else 6)  # The number of agents in the environment
        self._window_size = 750  # The size of the PyGame window

        self.agents = [str(chr(r + 65)) for r in range(self._num_agents)]
        self.possible_agents = self.agents[:]

        # Initialize the grid
        self._grid = Grid(self._rows, self._columns)

        # We have 2 * rows * columns + rows + columns actions, corresponding to closing each edge
        self._horizontal_edges = (self._rows + 1) * self._columns
        self._vertical_edges = self._rows * (self._columns + 1)
        self._num_edges = self._horizontal_edges + self._vertical_edges

        self.action_spaces = {i: spaces.Discrete(
            self._num_edges) for i in self.agents}

        # Observations are a Box of size (rows, columns)
        # Each element of the box is an integer in {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
        # The integer encodes which edges are closed. For example, if the integer is 3, then
        # the binary representation of 3 is 0011, which means that the bottom and left edges
        # are closed. The order of edges in the binary representation is top, right, bottom, left.
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    # "observation": spaces.Box(
                    #     low=0, high=15, shape=(self._rows, self._columns), dtype=np.int8
                    # ),
                    "observation": spaces.Box(low=0, high=1, shape=(self._num_edges,), dtype=np.int8),
                    "action_mask": spaces.Box(low=0, high=1, shape=(self._num_edges,), dtype=np.int8),
                }
            )
            for i in self.agents
        }

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self._starting_agent_index = 0

        self.window = None
        self.clock = pygame.time.Clock()

    def observe(self, agent):
        observation = np.zeros(self._num_edges, dtype=np.int8)
        for i in range(self._num_edges):
            if not self._grid._is_action_valid(i):
                observation[i] = 1

        legal_moves = self._legal_moves() if agent == self.agent_selection else []

        action_mask = np.zeros(self._num_edges, dtype=np.int8)
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _legal_moves(self):
        return [i for i in range(self._num_edges) if self._grid._is_action_valid(i)]

    # action in this case is a value from 0 to 6 indicating position to move on the flat representation of the connect4 board
    def step(self, action):
        if (
            self.truncations[self.agent_selection]
            or self.terminations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        # assert valid move
        # assert self._grid._is_action_valid(action), "played illegal move."

        action_valid, first_cell, second_cell, first_cell_closed_after_action, second_cell_closed_after_action = self._grid._apply_action(
            action)

        # If the agent has closed both adjacent cells, we give it a reward of 2
        # If the agent has closed only one cell, we give it a reward of 1
        # If the agent has not closed any cell, we give it a reward of 0
        cells_closed = 2 if (
            first_cell_closed_after_action and second_cell_closed_after_action
        ) else 1 if (
            first_cell_closed_after_action or second_cell_closed_after_action
        ) else 0

        # If the agent has closed a cell, we update the cell_belongs_to array
        if first_cell_closed_after_action:
            self._cell_belongs_to[first_cell[0]
                                  ][first_cell[1]] = self.agent_selection
        if second_cell_closed_after_action:
            self._cell_belongs_to[second_cell[0]
                                  ][second_cell[1]] = self.agent_selection

        # If the agent has closed a cell, we give it another turn
        if cells_closed > 0:
            next_agent = self.agent_selection
        else:
            next_agent = self._agent_selector.next()

        # Update the last action, last player, last reward - used for rendering
        self._last_action = action
        self._last_player = self.agent_selection
        self._last_reward = cells_closed

        if not action_valid:
            # If the agent has played an illegal move, we give it a reward of -1
            self.rewards = {i: -10 if i ==
                            self.agent_selection else 0 for i in self.agents}
        elif self._grid._check_if_all_edges_closed():
            # If the game is over, we give the agent a reward of 1 if it has the most cells, otherwise we give it a reward of 0 if game is a tie else -1
            winning_agent = self._get_winning_agent()
            winning_score = self._get_agent_score(winning_agent)
            self.rewards = {i: 0 if self._get_agent_score(i) < winning_score else (
                0 if self._game_tied() else 1) for i in self.agents}
        else:
            self.rewards = {i: 0 for i in self.agents}

        # self.rewards[self.agent_selection] += (
        #    cells_closed / (self._rows * self._columns))

        # Alternate reward system where agents get a positive or negative reward in each step depending upon cells closed
        # self.rewards = {i: cells_closed if i == self.agent_selection else ((0 - cells_closed)/(self.num_agents - 1)) for i in self.agents}

        game_over = self._grid._check_if_all_edges_closed() or not action_valid
        self.terminations = {i: game_over for i in self.agents}
        self.agent_selection = next_agent

        self._accumulate_rewards()
        self._step_count += 1

        if self.render_mode == "human":
            self.render()

    def _game_tied(self):
        game_tied = False
        if self._grid._check_if_all_edges_closed():
            winning_agent = self._get_winning_agent()
            winning_score = self._get_agent_score(winning_agent)
            for i in self.agents:
                if i != winning_agent and self._get_agent_score(i) == winning_score:
                    game_tied = True
                    break
        return game_tied

    def _get_agent_score(self, agent):
        return np.count_nonzero(np.array(self._cell_belongs_to) == agent)

    def _get_winning_agent(self):
        winning_agent = None
        if self._grid._check_if_all_edges_closed():
            for i in self.agents:
                if winning_agent is None:
                    winning_agent = i
                elif self._get_agent_score(i) > self._get_agent_score(winning_agent):
                    winning_agent = i
        return winning_agent

    def _get_final_reward_for_agent(self, agent):
        agent_total_rewards = {i: 0 for i in self.agents}
        for i in self.agents:
            total = np.count_nonzero(np.array(self._cell_belongs_to) == i)
            agent_total_rewards[i] += total
            for j in self.agents:
                if i != j:
                    agent_total_rewards[j] -= (total/(self.num_agents-1))
        return agent_total_rewards[agent]

    def reset(self, render_mode=None, seed=None, options=None):

        if render_mode is not None:
            self.render_mode = render_mode

        self._grid = Grid(self._rows, self._columns)
        self.agents = self.possible_agents[:]
        self.rewards = {i: 0.0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        self.agent_selection = self.agents[self._starting_agent_index]
        self._starting_agent_index = (self._starting_agent_index + 1) % len(
            self.agents)
        while self._agent_selector.selected_agent != self.agent_selection:
            self._agent_selector.next()

        self._step_count = 0
        self._last_action = None
        self._last_player = None
        self._last_reward = 0
        self._cell_belongs_to = [
            ['' for _ in range(self._columns)] for _ in range(self._rows)]

        if self.render_mode == "human":
            self.render()

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self._window_size, self._window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self._window_size, self._window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self._window_size / (max(self._rows, self._columns) * 3)
        )  # The size of a single grid square in pixels

        x_offset = (self._window_size - pix_square_size * self._columns) / 2
        y_offset = (self._window_size - pix_square_size * self._rows) / 2

        # We draw the grid in four steps. First, we draw all the vertices of the grid.
        for x in range(self._rows + 1):
            for y in range(self._columns + 1):
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (x_offset + pix_square_size * x,
                     y_offset + pix_square_size * y),
                    5,
                )

        # Next, we draw all the horizontal edges
        for x in range(self._rows):
            for y in range(self._columns):
                if self._grid._check_edge_already_closed([x, y], "top"):
                    pygame.draw.line(
                        canvas,
                        0,
                        (x_offset + pix_square_size * y,
                         y_offset + pix_square_size * x),
                        (x_offset + pix_square_size * (y + 1),
                         y_offset + pix_square_size * x),
                        width=3,
                    )
            if x == self._rows - 1:
                for y in range(self._columns):
                    if self._grid._check_edge_already_closed([x, y], "bottom"):
                        pygame.draw.line(
                            canvas,
                            0,
                            (x_offset + pix_square_size * y,
                             y_offset + (pix_square_size * self._rows)),
                            (x_offset + pix_square_size * (y + 1),
                             y_offset + (pix_square_size * self._rows)),
                            width=3,
                        )

        # Next, we draw all the vertical edges
        for x in range(self._rows):
            for y in range(self._columns):
                if self._grid._check_edge_already_closed([x, y], "left"):
                    pygame.draw.line(
                        canvas,
                        0,
                        (x_offset + pix_square_size * y,
                         y_offset + pix_square_size * x),
                        (x_offset + pix_square_size * y,
                         y_offset + pix_square_size * (x + 1)),
                        width=3,
                    )
                if y == self._columns - 1:
                    if self._grid._check_edge_already_closed([x, y], "right"):
                        pygame.draw.line(
                            canvas,
                            0,
                            (x_offset + pix_square_size * self._columns,
                             y_offset + pix_square_size * x),
                            (x_offset + pix_square_size * self._columns,
                             y_offset + pix_square_size * (x + 1)),
                            width=3,
                        )

        # Finally, we label all the cells with the agent that owns them
        colorMap = {
            'A': (0, 255, 0),
            'B': (0, 0, 255),
            'C': (255, 0, 0),
            'D': (255, 255, 0),
            'E': (255, 0, 255),
            'F': (0, 255, 255),
        }
        for x in range(self._rows):
            for y in range(self._columns):
                if self._cell_belongs_to[x][y] != '':
                    font = pygame.font.SysFont("Arial", 30)
                    text = font.render(
                        str(self._cell_belongs_to[x][y]),
                        True,
                        colorMap[self._cell_belongs_to[x][y]],
                    )
                    canvas.blit(
                        text,
                        (
                            x_offset + pix_square_size * y + pix_square_size / 2 - text.get_width() / 2,
                            y_offset + pix_square_size * x + pix_square_size / 2 - text.get_height() / 2,
                        ),
                    )

        if self._last_action is not None:
            # Display the last player, last action taken and last reward below the grid
            font = pygame.font.SysFont("Arial", 30)
            text = font.render(
                "Player: " +
                str(self._last_player if self._last_player is not None else None),
                True,
                colorMap[self._last_player],
            )
            canvas.blit(
                text,
                (
                    x_offset + pix_square_size * self._columns / 2 - text.get_width() / 2,
                    y_offset + pix_square_size * (self._rows + 1),
                ),
            )
            action_cells = self._grid._get_cells_from_action(
                self._last_action) if self._last_action is not None else None
            text = font.render(
                "Action taken: Edge between " +
                str(action_cells[0]) +
                " and " +
                str(action_cells[1]),
                True,
                (0, 0, 0),
            )
            canvas.blit(
                text,
                (
                    x_offset + pix_square_size * self._columns / 2 - text.get_width() / 2,
                    y_offset + pix_square_size *
                    (self._rows + 1) + text.get_height(),
                ),
            )
            text = font.render(
                "Reward: " + str(self._last_reward),
                True,
                (0, 0, 0),
            )
            canvas.blit(
                text,
                (
                    x_offset + pix_square_size * self._columns / 2 - text.get_width() / 2,
                    y_offset + pix_square_size *
                    (self._rows + 1) + 2 * text.get_height(),
                ),
            )

        # Display the next player in the top right corner
        font = pygame.font.SysFont("Arial", 30)
        text = font.render(
            "Next Turn: " + str(self.agent_selection),
            True,
            colorMap[self.agent_selection],
        )
        canvas.blit(text, (x_offset + pix_square_size * self._columns, 0))

        # Display the step count in the top left corner
        font = pygame.font.SysFont("Arial", 30)
        text = font.render("Steps finished: " +
                           str(self._step_count), True, (0, 0, 0))
        canvas.blit(text, (0, 0))

        # Display the scores of each player above the grid
        font = pygame.font.SysFont("Arial", 30)
        for i in range(self._num_agents):
            text = font.render(
                "Player " + str(chr(i + 65)) + ": " +
                str(self._cumulative_rewards[str(chr(i + 65))]),
                True,
                colorMap[str(chr(i + 65))],
            )
            canvas.blit(
                text,
                (
                    x_offset + pix_square_size * self._columns / 2 - text.get_width() / 2,
                    y_offset - ((text.get_height() * 1.1) *
                                (self._num_agents - i + 1)),
                ),
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
