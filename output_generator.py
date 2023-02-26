import csv
import os
import time

import numpy as np


class OutputGenerator:
    """Class to generate csv-files of the simulation results (pedestrian and vehicle trajectories)"""

    def __init__(self, ped_sim, output_path, scenario_name):
        self.scene = ped_sim
        self.ped_states = self.scene.peds.all_states
        self.veh_states = self.scene.all_dyn_obs_states
        self.static_obstacles = self.scene.static_obstacles
        self.borders = self.scene.borders
        self.output_path = output_path

        time_stamp = time.strftime('%Y%m%d-%H%M%S')

        if scenario_name:
            dir_name = time_stamp + '-' + scenario_name
        else:
            dir_name = time_stamp

        self.output_dir = os.path.join(output_path, dir_name)

        # If folders path does not exist, create it
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_ped_csv(self):
        output_file = os.path.join(self.output_dir, 'pedestrian.csv')

        header = ['ped_id', 'frame', 'time', 'x', 'y', 'v_x', 'v_y', 'mode']

        with open(output_file, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            writer.writerow(header)

            for frame, (sim_time, state) in enumerate(self.ped_states.items()):
                for ped in state:
                    ped_id = int(ped['name'].split('_')[-1])
                    x = ped['loc'][0]
                    y = ped['loc'][1]
                    v_x = ped['vel'][0]
                    v_y = ped['vel'][1]
                    mode = ped['mode']

                    writer.writerow([ped_id, frame, sim_time, x, y, v_x, v_y, mode])

    def generate_veh_csv(self):
        output_file = os.path.join(self.output_dir, 'vehicle.csv')

        header = ['veh_id', 'frame', 'time', 'x', 'y', 'heading', 'vel', 'ext_x', 'ext_y']

        with open(output_file, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            writer.writerow(header)

            for frame, (sim_time, state) in enumerate(self.veh_states.items()):
                for veh in state:
                    veh_id = veh['id']
                    x = veh['loc'][0]
                    y = veh['loc'][1]
                    heading = np.deg2rad(veh['heading'])
                    vel = np.linalg.norm(veh['vel'])
                    ext_x = veh['extent'][0]
                    ext_y = veh['extent'][1]

                    writer.writerow([veh_id, frame, sim_time, x, y, heading, vel, ext_x, ext_y])

    def generate_borders_csv(self):
        output_file = os.path.join(self.output_dir, 'borders.csv')

        header = ['x', 'y']

        with open(output_file, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            writer.writerow(header)

            for border in self.borders:
                for point in border:
                    x = point[0]
                    y = point[1]

                    writer.writerow([x, y])

    def generate_obstacles_csv(self):
        output_file = os.path.join(self.output_dir, 'obstacles.csv')

        header = ['obs_id', 'obs_pos_x', 'obs_pos_y', 'x', 'y']

        with open(output_file, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            writer.writerow(header)

            for obs_id, (pos, border) in enumerate(self.static_obstacles):
                x_pos = pos[0]
                y_pos = pos[1]

                for point in border:
                    x = point[0]
                    y = point[1]

                    writer.writerow([obs_id, x_pos, y_pos, x, y])
