"""Utility functions for plots and animations."""
import logging

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

from stateutils import minmax


class SceneVisualizer:
    """Context for social force visualization"""

    def __init__(self, ped_sim, output, step_length, writer='pillow', cmap='viridis', agent_colors=None, **kwargs):
        self.scene = ped_sim
        self.states = self.scene.get_states()
        self.cmap = cmap
        self.agent_colors = agent_colors
        self.frames = len(self.states)
        self.output = output
        self.step_length = step_length
        self.writer = writer

        self.fig, self.ax = plt.subplots(**kwargs)

        self.ani = None

        self.human_actors = None
        self.human_collection = PatchCollection([])
        self.human_collection.set(animated=True, alpha=0.6, cmap=self.cmap, clip_on=True)

    def plot(self):
        """Main method to create plot"""
        self.plot_borders()

        for ped in range(self.scene.peds.size()):
            x = self.states[:, ped]['loc'][:, 0]
            y = self.states[:, ped]['loc'][:, 1]
            name = self.states[0, ped]['name']
            self.ax.plot(x, y, '-', label=name)
            self.ax.arrow(x[0], y[0], x[10] - x[0], y[10] - y[0], width=0.1, zorder=10)

        self.ax.legend(bbox_to_anchor=(0.5, 1.04), loc='lower center', ncol=5)
        return self.fig

    def animate(self):
        """Main method to create animation"""

        self.ani = mpl_animation.FuncAnimation(
            self.fig,
            init_func=self.animation_init,
            func=self.animation_update,
            frames=self.frames,
            interval=0,
            blit=False,
        )

        return self.ani

    def __enter__(self):
        logging.info('Start plotting.')
        self.fig.set_tight_layout(True)
        self.ax.grid(linestyle='dotted')
        self.ax.set_aspect('equal')
        self.ax.margins(2.0)
        self.ax.set_axisbelow(True)
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')

        plt.rcParams['animation.html'] = 'jshtml'

        # x, y limit from states, only for animation
        margin = 2.0
        xy_limits = np.array(
            [minmax(state) for state in self.states]
        )  # (x_min, y_min, x_max, y_max)
        xy_min = np.min(xy_limits[:, :2], axis=0) - margin
        xy_max = np.max(xy_limits[:, 2:4], axis=0) + margin
        # invert y-axis by switching y_min and y_max, because CARLA uses left-hand coordinate system
        self.ax.set(xlim=(xy_min[0], xy_max[0]), ylim=(xy_max[1], xy_min[1]))

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type:
            logging.error(
                f'Exception type: {exception_type}; Exception value: {exception_value}; Traceback: {traceback}'
            )
        if self.output:
            if self.ani:
                output = self.output + '.gif'
                logging.info(f'Saving animation as {output}')
                fps = int(1 / self.step_length)
                self.ani.save(output, writer=self.writer, fps=fps)
            else:
                output = self.output + '.png'
                logging.info(f'Saving plot as {output}')
                self.fig.savefig(output, dpi=300)
        plt.close(self.fig)
        logging.info('Plotting ends.')

    def plot_human(self, step=-1):
        """Generate patches for human
        :param step: index of state, default is the latest
        :return: list of patches
        """
        current_state = self.states[step]
        radius = [0.2] * current_state.shape[0]
        if self.human_actors:
            for i, human in enumerate(self.human_actors):
                human.center = current_state[i]['loc']
                human.set_radius(0.2)
        else:
            self.human_actors = [
                Circle(pos, radius=r) for pos, r in zip(current_state[:]['loc'], radius)
            ]
        self.human_collection.set_paths(self.human_actors)
        if not self.agent_colors:
            self.human_collection.set_array(np.arange(current_state.shape[0]))
        else:
            # set colors for each agent
            assert len(self.human_actors) == len(
                self.agent_colors
            ), 'agent_colors must be the same length as the agents'
            self.human_collection.set_facecolor(self.agent_colors)

    def plot_borders(self):
        borders = self.scene.get_borders()

        if borders is not None:
            for o in borders:
                self.ax.plot(o[:, 0], o[:, 1], lw=2, color='black')

    def animation_init(self):
        self.plot_borders()
        self.ax.add_collection(self.human_collection)

        return self.human_collection,

    def animation_update(self, i):
        self.plot_human(i)
        return self.human_collection,
