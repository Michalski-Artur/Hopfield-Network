import matplotlib.pyplot as plt
import pygame
import numpy as np


class DataVisualizer:

    max_dimension = 600
    cell_size = 30
    active_neuron_color = (0, 0, 128)
    inactive_neuron_color = (135, 206, 250)
    other_state_neuron_color = (255, 140, 0)
    surface_color = (211, 211, 211)
    timeframe = 100
    neurons_updated_in_frame = 16

    @staticmethod
    def visualize_convergence(hopfield_network, single_pattern_size):
        # initialize pygame
        pygame.init()
        # set dimensions of board and cellsize
        # window size: single_pattern_size[0] X single_pattern_size[1]
        DataVisualizer.cell_size = max(DataVisualizer.max_dimension / single_pattern_size[1], DataVisualizer.max_dimension / single_pattern_size[0])
        surface = pygame.display.set_mode((single_pattern_size[0] * DataVisualizer.cell_size,
                                           single_pattern_size[1] * DataVisualizer.cell_size))
        pygame.display.set_caption("   ")

        # kill pygame if user exits window
        running = True
        # main animation loop
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                    # quit pygame
                    pygame.quit()
                    # plt.show()
                    return

            cells = hopfield_network.neurons_state.reshape(single_pattern_size[0], single_pattern_size[1]).T

            # fills surface with color
            surface.fill(DataVisualizer.surface_color)

            # loop through network state array and update colors for each cell
            for r, c in np.ndindex(cells.shape):  # iterates through all cells in cells matrix
                if cells[r, c] == -1:
                    col = DataVisualizer.inactive_neuron_color

                elif cells[r, c] == 1:
                    col = DataVisualizer.active_neuron_color

                else:
                    col = DataVisualizer.other_state_neuron_color
                pygame.draw.rect(surface, col, (r * DataVisualizer.cell_size, c * DataVisualizer.cell_size,
                                                DataVisualizer.cell_size, DataVisualizer.cell_size))  # draw new cell_

            # update network state
            hopfield_network.update_state(DataVisualizer.neurons_updated_in_frame)

            # updates display from new .draw in update function
            pygame.display.update()
            pygame.time.wait(DataVisualizer.timeframe)

    @staticmethod
    def visualize_pattern(pattern):
        # Conversion from bipolar to unipolar
        pattern = (pattern + 1) / 2
        plt.figure("pattern", figsize=pattern.shape, dpi=DataVisualizer.cell_size)
        plt.imshow(pattern, cmap='RdPu')
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.show()

