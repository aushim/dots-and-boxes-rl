import numpy as np

class Grid:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.horizontal_edges = (rows + 1) * columns
        self.vertical_edges = rows * (columns + 1)
        self.num_edges = self.horizontal_edges + self.vertical_edges
        self._grid = np.zeros((rows, columns), dtype=np.uint8)

    def _get_cells_from_action(self, action):
        self._action_to_cells = {x: [[int(x/self.columns) - 1, x%self.columns], [int(x/self.columns), x%self.columns]] for x in range(self.num_edges) if x < self.horizontal_edges}
        self._action_to_cells.update({x: [[int((x - self.horizontal_edges)/(self.columns + 1)), (x - self.horizontal_edges)%(self.columns + 1) - 1], [int((x - self.horizontal_edges)/(self.columns + 1)), (x - self.horizontal_edges)%(self.columns + 1)]] for x in range(self.num_edges) if x >= self.horizontal_edges})

        return self._action_to_cells[action] if action < self.num_edges else None
    
    def _get_cell_and_edge_type_from_action(self, action):
        if action < self.num_edges:
            cells = self._get_cells_from_action(action)
            if action < self.horizontal_edges:
                if self._check_cell_in_bounds(cells[0]):
                    return cells[0], 'bottom'
                else:
                    return cells[1], 'top'
            else:
                if self._check_cell_in_bounds(cells[0]):
                    return cells[0], 'right'
                else:
                    return cells[1], 'left'
        else:
            return None, None

    def _is_action_valid(self, action):
        cell, edge_type = self._get_cell_and_edge_type_from_action(action)

        return not self._check_edge_already_closed(cell, edge_type)

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
    
    def _apply_action(self, action):
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

        return action_valid, first_cell, second_cell, first_cell_closed_after_action, second_cell_closed_after_action