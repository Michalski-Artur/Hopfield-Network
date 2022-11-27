import matplotlib.pyplot as plt

class DataVisualizer:
    MAX_DIMENSION = 2500

    def __init__(self, single_pattern_size, target_pattern, input_pattern, figure_title = '', output_path = None):
        self.__figure_title = figure_title
        self.__dpi = self.MAX_DIMENSION / (3*single_pattern_size[1])
        self.__font_size = 1000 / self.__dpi
        self.__single_pattern_size = single_pattern_size
        self.__figsize = (single_pattern_size[1], single_pattern_size[0])
        self.__target_pattern_unipolar = (target_pattern + 1) / 2
        self.__input_pattern_unipolar = (input_pattern + 1) / 2
        self.__output_path = output_path
        self.__figure = None
        self.__axis = None

    def visualize_step(self, current_state, step_number, save_figure = False):
        current_state_unipolar = (current_state + 1) / 2
        if self.__figure is None or self.__axis is None or not plt.fignum_exists(self.__figure.number):
            self.__figure, self.__axis = plt.subplots(1, 3, figsize=self.__figsize, dpi=self.__dpi)
            self.__figure.suptitle(self.__figure_title, fontsize=2*self.__font_size)
        self.__axis[0].imshow(self.__target_pattern_unipolar.reshape(self.__single_pattern_size), cmap='RdPu')
        self.__axis[0].set_title('Target pattern', fontsize=self.__font_size)
        self.__axis[1].imshow(self.__input_pattern_unipolar.reshape(self.__single_pattern_size), cmap='RdPu')
        self.__axis[1].set_title('Input (noised) pattern', fontsize=self.__font_size)
        self.__axis[2].imshow(current_state_unipolar.reshape(self.__single_pattern_size), cmap='RdPu')
        self.__axis[2].set_title(f'Step {step_number} of Hopfield network', fontsize=self.__font_size)

        plt.draw()
        plt.pause(0.1)
        if save_figure and self.__output_path is not None:
            self.__figure.savefig(f'{self.__output_path}_step{step_number}.png')
            plt.close(self.__figure)
