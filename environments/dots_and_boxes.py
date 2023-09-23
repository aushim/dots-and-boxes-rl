import gym
from gym import spaces
import pygame
import numpy as np


class DotsAndBoxesEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, rows=5, columns=5, num_agents=3):
        self.rows = rows if rows <=10 else 10 # The number of rows in the grid
        self.columns = columns  if columns <=10 else 10 # The number of columns in the grid
        self.num_agents = num_agents if num_agents <=6 else 6  # The number of agents in the environment
        self.window_size = 1000  # The size of the PyGame window

        # Observations are a Box of size (rows, columns)
        # Each element of the box is an integer in {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
        # The integer encodes which edges are closed. For example, if the integer is 3, then
        # the binary representation of 3 is 0011, which means that the bottom and left edges
        # are closed. The order of edges in the binary representation is top, right, bottom, left.
        self.observation_space = spaces.Box(0, 15, shape=(rows, columns), dtype=int)
 
        # We have 2 * rows * columns + rows + columns actions, corresponding to closing each edge
        self.horizontal_edges = (rows + 1) * columns
        self.vertical_edges = rows * (columns + 1)
        self.num_edges = self.horizontal_edges + self.vertical_edges
        self.action_space = spaces.Discrete(self.num_edges)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the adjacent cells whose edge we will close if that action is taken.
        """
        self._action_to_cells = {x: [[int(x/columns) - 1, x%columns], [int(x/columns), x%columns]] for x in range(self.num_edges) if x < self.horizontal_edges}
        self._action_to_cells.update({x: [[int((x - self.horizontal_edges)/(columns + 1)), (x - self.horizontal_edges)%(columns + 1) - 1], [int((x - self.horizontal_edges)/(columns + 1)), (x - self.horizontal_edges)%(columns + 1)]] for x in range(self.num_edges) if x >= self.horizontal_edges})

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
            "grid": self._grid,
            "current_player": self._current_player
        }

    def _get_info(self):
        return {
            "cell_belongs_to": self._cell_belongs_to,
            "current_player": self._current_player,
            "last_player": self._last_player,
            "last_action": self._last_action,
            "last_reward": self._last_reward,
            "step_count": self._step_count
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Create a new grid
        self._grid = np.zeros((self.rows, self.columns), dtype=int)
        self._current_player = 0
        self._last_player = None
        self._last_action = None
        self._last_reward = 0
        self._step_count = 0

        # cell_belongs_to is an array of size (rows, columns)
        # Each element of the array is an integer in {0,1,2,...,num_agents - 1}
        # It encodes which agent owns the cell. For example, if the integer is 3, then
        # the cell belongs to the 3rd agent.
        self._cell_belongs_to = np.full((self.rows, self.columns), -1, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _check_cell_in_bounds(self, cell):
        return cell[0] >= 0 and cell[0] < self.rows and cell[1] >= 0 and cell[1] < self.columns
    
    def _check_edge_already_closed(self, cell, edge_type):
        if self._check_cell_in_bounds(cell):
            if edge_type == 'left':
                return self._grid[cell[0]][cell[1]] & 1 == 1
            elif edge_type == 'bottom':
                return self._grid[cell[0]][cell[1]] & 2 == 2
            elif edge_type == 'right':
                return self._grid[cell[0]][cell[1]] & 4 == 4
            elif edge_type == 'top':
                return self._grid[cell[0]][cell[1]] & 8 == 8
        
        return False
    
    def _close_cell_edge(self, cell, edge_type):
        cell_closed_after_action = False

        if self._check_cell_in_bounds(cell):
            # check if the cell is already closed
            if self._grid[cell[0]][cell[1]] == 15:
                return False
            
            # close the edge
            if edge_type == 'left':
                self._grid[cell[0]][cell[1]] |= 1
            elif edge_type == 'bottom':
                self._grid[cell[0]][cell[1]] |= 2
            elif edge_type == 'right':
                self._grid[cell[0]][cell[1]] |= 4
            elif edge_type == 'top':
                self._grid[cell[0]][cell[1]] |= 8

            # check if the cell is now closed
            cell_closed_after_action = self._grid[cell[0]][cell[1]] == 15
        
        return cell_closed_after_action
    
    def _check_if_all_edges_closed(self):
        for row in self._grid:
            for cell in row:
                if cell != 15:
                    return False
        
        return True    

    def step(self, action):
        self._last_player = self._current_player
        self._last_action = action
        self._step_count += 1

        # Map the action (element of {0,1,2,3...,num_edges - 1}) to the adjacent cells whose edge we will close
        edge_type = 'horizontal' if action < self.horizontal_edges else 'vertical'
        [first_cell, second_cell] = self._action_to_cells[action]
        action_valid = True
        
        first_cell_closed_after_action = False
        second_cell_closed_after_action = False

        if edge_type == 'horizontal':
            [top_cell, bottom_cell] = [first_cell, second_cell]
            action_valid = not self._check_edge_already_closed(top_cell, 'bottom') and not self._check_edge_already_closed(bottom_cell, 'top')
            if action_valid:
                first_cell_closed_after_action = self._close_cell_edge(top_cell, 'bottom')
                second_cell_closed_after_action = self._close_cell_edge(bottom_cell, 'top')

        if edge_type == 'vertical':
            [left_cell, right_cell] = [first_cell, second_cell]
            action_valid = not self._check_edge_already_closed(left_cell, 'right') and not self._check_edge_already_closed(right_cell, 'left')
            if action_valid:
                first_cell_closed_after_action = self._close_cell_edge(left_cell, 'right')
                second_cell_closed_after_action = self._close_cell_edge(right_cell, 'left')

        if action_valid:
            # If the agent has closed both adjacent cells, we give it a reward of 2
            # If the agent has closed only one cell, we give it a reward of 1
            # If the agent has not closed any cell, we give it a reward of 0
            reward = 2 if (
                    first_cell_closed_after_action and second_cell_closed_after_action
                ) else 1 if (
                    first_cell_closed_after_action or second_cell_closed_after_action
                ) else 0
            
            # If the agent has closed a cell, we update the cell_belongs_to array
            if first_cell_closed_after_action:
                self._cell_belongs_to[first_cell[0]][first_cell[1]] = self._current_player
            if second_cell_closed_after_action:
                self._cell_belongs_to[second_cell[0]][second_cell[1]] = self._current_player

            # If the agent has not closed a cell, we switch the current player
            if reward == 0:
                self._current_player = (self._current_player + 1) % self.num_agents
        else:
            # If the action is invalid i.e. the edge is already closed, we return a reward of 0 and the player has to take another action
            # We do not switch the current player
            reward = -1
        
        self._last_reward = reward

        terminated = self._check_if_all_edges_closed()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / (max(self.rows, self.columns) * 3)
        )  # The size of a single grid square in pixels

        x_offset = (self.window_size - pix_square_size * self.columns) / 2
        y_offset = (self.window_size - pix_square_size * self.rows) / 2

        # We draw the grid in four steps. First, we draw all the vertices of the grid.
        for x in range(self.rows + 1):
            for y in range(self.columns + 1):
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (x_offset + pix_square_size * x, y_offset + pix_square_size * y),
                    5,
                )

        # Next, we draw all the horizontal edges
        for x in range(self.rows):
            for y in range(self.columns):
                if self._grid[x][y] & 8 == 8:
                    pygame.draw.line(
                        canvas,
                        0,
                        (x_offset + pix_square_size * y, y_offset + pix_square_size * x),
                        (x_offset + pix_square_size * (y + 1), y_offset + pix_square_size * x),
                        width=3,
                    )
            if x == self.rows - 1:
                for y in range(self.columns):
                    if self._grid[x][y] & 2 == 2:
                        pygame.draw.line(
                            canvas,
                            0,
                            (x_offset + pix_square_size * y, y_offset + (pix_square_size * self.rows)),
                            (x_offset + pix_square_size * (y + 1), y_offset + (pix_square_size * self.rows)),
                            width=3,
                        )

        # Next, we draw all the vertical edges
        for x in range(self.rows):
            for y in range(self.columns):
                if self._grid[x][y] & 1 == 1:
                    pygame.draw.line(
                        canvas,
                        0,
                        (x_offset + pix_square_size * y, y_offset + pix_square_size * x),
                        (x_offset + pix_square_size * y, y_offset + pix_square_size * (x + 1)),
                        width=3,
                    )
                if y == self.columns - 1:
                    if self._grid[x][y] & 4 == 4:
                        pygame.draw.line(
                            canvas,
                            0,
                            (x_offset + pix_square_size * self.columns, y_offset + pix_square_size * x),
                            (x_offset + pix_square_size * self.columns, y_offset + pix_square_size * (x + 1)),
                            width=3,
                        )

        # Finally, we label all the cells with the agent that owns them
        colorMap = {
            0: (0, 255, 0),
            1: (0, 0, 255),
            2: (255, 0, 0),
            3: (255, 255, 0),
            4: (255, 0, 255),
            5: (0, 255, 255),
        }
        for x in range(self.rows):
            for y in range(self.columns):
                if self._cell_belongs_to[x][y] != -1:
                    font = pygame.font.SysFont("Arial", 30)
                    text = font.render(
                        str(chr(self._cell_belongs_to[x][y] + 65)),
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
                "Player: " + str(chr(self._last_player + 65) if self._last_player is not None else None),
                True,
                colorMap[self._last_player],
            )
            canvas.blit(
                text,
                (
                    x_offset + pix_square_size * self.columns / 2 - text.get_width() / 2,
                    y_offset + pix_square_size * (self.rows + 1),
                ),
            )
            action_cells = self._action_to_cells[self._last_action] if self._last_action is not None else None
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
                    x_offset + pix_square_size * self.columns / 2 - text.get_width() / 2,
                    y_offset + pix_square_size * (self.rows + 1) + text.get_height(),
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
                    x_offset + pix_square_size * self.columns / 2 - text.get_width() / 2,
                    y_offset + pix_square_size * (self.rows + 1) + 2 * text.get_height(),
                ),
            )

        # Display the next player in the top right corner
        font = pygame.font.SysFont("Arial", 30)
        text = font.render(
            "Next Turn: " + str(chr(self._current_player + 65)),
            True,
            colorMap[self._current_player],
        )
        canvas.blit(text, (x_offset + pix_square_size * self.columns, 0))

        # Display the step count in the top left corner
        font = pygame.font.SysFont("Arial", 30)
        text = font.render("Steps finished: " + str(self._step_count), True, (0, 0, 0))
        canvas.blit(text, (0, 0))

        # Display the scores of each player above the grid
        font = pygame.font.SysFont("Arial", 30)
        for i in range(self.num_agents):
            text = font.render(
                "Player " + str(chr(i + 65)) + ": " + str(np.sum(self._cell_belongs_to == i)),
                True,
                colorMap[i],
            )
            canvas.blit(
                text,
                (
                    x_offset + pix_square_size * self.columns / 2 - text.get_width() / 2,
                    y_offset - ((text.get_height() * 1.1) * (self.num_agents - i + 1)),
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
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
