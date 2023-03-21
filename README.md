# CARLA Social Force Model


This project implements the pedestrian Social Force Model (SFM) based on Moussaïd et al. [[1]](#1) and couples it with 
the CARLA simulator via its Python API.

In order to make the SFM suitable for the simulation of urban traffic scenarios, known issues of the original model,
like pedestrians getting stuck at small obstacles, were improved and new features that enable modeled pedestrians to 
navigate independently and realistically in an urban environment were added.


## Features

#### Pedestrian model:
- Realistic pedestrian to pedestrian interaction based on the SFM
- Improved interaction and collision avoidance with static and dynamic obstacles (vehicles, bikes) with new obstacle interaction force
- Global routing via automatically generated navigation graph from CARLA maps (with or without jaywalking)
- Automatic sidewalk border extraction from CARLA maps to restrict pedestrians to walking on sidewalks via social forces
- Gap acceptance model that lets pedestrians check for crossing traffic before crossing a road


## Setup
1. Install Python (recommended Python version is 3.8)
2. Install CARLA according to its [documentation](https://carla.readthedocs.io/en/latest/start_quickstart/) 
(Version 0.9.13 is recommended but newer versions should work as well)
3. Check out this repository
4. Install external dependencies with:
   ```sh
   pip install -r requirements.txt
   ```
5. Add `<path-to-carla-installation>/carla-simulator/PythonAPI/carla` to your PYTHONPATH

## Usage
1. Start CARLA server with:
   ```sh
   ./CarlaUE4.sh
   ```
2. Run CARLA client with pedestrian simulation with:
   ```sh
   python run_simulation.py 
   ```

## Configuration

### SFM configuration

The parameters of the SFM can be configured in the `config/sfm_config.toml` file.
This includes the parameters of the different social forces as well as which forces are active during simulation.

```TOML
# Configuration file for all parameters of the used Social Force Model 

max_speed_multiplier = 1.3          # multiplier that defines how much faster the maximum resulting speed calculated by
                                    # the SFM is allowed to be compared to the individually defined target speed
use_ped_radius = false              # consider pedestrian radius in social forces calculations

# activate of deactivate different social forces
[forces]
acceleration_force = true
pedestrian_force = true
border_force = true
static_obstacle_force = true
dynamic_obstacle_force = true

# acceleration force parameters
[acceleration_force]
tau = 0.5       # relaxation time within which the target velocity should be reached

# pedestrian interaction force parameters
[pedestrian_force]
lambda = 2.0    # factor influencing the weight between direction of relative motion and the interaction direction
A = 4.5         # amplitude of the resulting social force
gamma = 0.35    # parameter influencing how fast the resulting force decays with increasing distance
n = 2.0         # parameter influencing the angular interaction range taken into account for the partial force
                # responsible for the directional change
n_prime = 3.0   # parameter influencing the angular interaction range taken into account for the partial force
                # responsible for the deceleration
epsilon = 0.005 # interaction angle offset creating a bias either evade left or right

# border force parameters
[border_force]
a = 6.0         # amplitude of the border force
b = 0.3         # parameter influencing how fast the resulting force decays with increasing distance

# static obstacle interaction force parameters
[static_obstacle_force]
lambda = 2.3    # factor influencing the weight between direction of relative motion and the interaction direction
A = 15          # amplitude of the resulting social force
gamma = 0.4     # parameter influencing how fast the resulting force decays with increasing distance
n = 2.1         # parameter influencing the angular interaction range taken into account for the partial force
                # responsible for the directional change
n_prime = 3.0   # parameter influencing the angular interaction range taken into account for the partial force
                # responsible for the deceleration
epsilon = 0.005 # interaction angle offset creating a bias either evade left or right
perception_threshold = 20   # threshold determining how close obstacles need to be in order to be considered by SFM

# dynamic obstacle interaction force parameters
[dynamic_obstacle_force]
lambda = 2.0    # factor influencing the weight between direction of relative motion and the interaction direction
A = 50          # amplitude of the resulting social force
gamma = 0.4     # parameter influencing how fast the resulting force decays with increasing distance
n = 1.0         # parameter influencing the angular interaction range taken into account for the partial force
                # responsible for the directional change
n_prime = 3.0   # parameter influencing the angular interaction range taken into account for the partial force
                # responsible for the deceleration
epsilon = 0.005 # interaction angle offset creating a bias either evade left or right
perception_threshold = 50   # threshold determining how close obstacles need to be in order to be considered by SFM
```

### Scenario configuration

The simulation scenario can be defined and configured with a `scenario_config.toml` file.

```TOML
scenario_name = 'example' # scenario name for output files

# map configurations
[map]
map_name = 'Town10HD_Opt'                 # CARLA map name
map_path = 'Carla/Maps/'                  # path to CARLA map from CarlaUE4/Content
draw_obstacles = false                    # draw (obstacle) border points (default = false)
unload_props = false                      # unload props like trees, benches etc. (default = false)
spectator_location = [75, 171.9, 10.0]    # location of spectator camera [x,y,z]
spectator_rotation = [-40.0, 0.0, 0.0]    # rotation of spectator camera [pitch,yaw,roll]

# general pedestrian configurations
[walker]
pedestrian_seed = 2020      # random seed for all random operations regarding pedestrians (default = 2000)
despawn_on_arrival = true   # despawn pedestrians on arrival (default = false)
waypoint_threshold = 1      # threshold [m] how close pedestrians have to get to a waypoint to consider them arrived
                            # (default = 2.0)
waypoint_distance = 5       # distance between waypoints in pedestrian navigation graph (default = 5.0)
jaywalking_weight = 2       # factor how much more expensive jaywalking edges are compared to normal edges for routing
                            # algorithm (default = 2.0)
draw_bounding_boxes = false # draw bounding boxes around pedestrians (default = false)
variate_speed = 0.1         # variate pedestrian speed in range +- [m/s] (default = 0.0)
random_pedestrians = 10     # number of random pedestrians with random destinations spawned across the map (default = 0)

# definition of pedestrian spawner(s)
[[walker.ped_spawner]]
spawn_location = [84.3, 171.9, 1.0]     # spawn location of spawner [x,y,z]
generate_route = 'JAYWALKING'           # generate route to destination with JAYWALKING, JAYWALKING_AT_JUNCTION or
                                        # NO_JAYWALKING (if not defined pedestrians head straight to their destination
                                        # without routing)
destination = [96.2, 172.2, 0.0]        # destination of spawner [x,y,z]
blueprint = 'walker.pedestrian.0002'    # CARLA blueprint of pedestrian (default = random)
speed = 1.5                             # target speed of pedestrian [m/s] (default = 1.2)
quantity = 1                            # number of pedestrians spawned from this spawner (default = 1)
spawn_time = 0.0                        # spawn time [s] of first pedestrian spawning with this spawner (default = 0.0)
spawn_interval = 2.8                    # time interval [s] between spawning of pedestrians (default = 3.0)
crossing_speed_factor = 1               # factor how the target speed changes when crossing a road [m/s] (default = 1.5)
crossing_safety_margin = 1              # safety margin [s] for gap acceptance model before crossing (default = 1.5)


# general vehicle configurations
[vehicle]
vehicle_seed = 100          # random seed for all random operations regarding vehicles (default = 2000)
variate_speed_factor = 3    # variate speed reduction factor +- [%] (default = 0.0)
no_bikes = true             # allow no bikes when selecting random blueprint (default = false)

# definition of vehicle spawner(s)
[[vehicle.vehicle_spawner]]
spawn_point = 103                   # CARLA spawn point number for vehicles (refers to a location of the road)
destination = 90                    # CARLA destination number for vehicles (refers to a location of the road, works
                                    # only with vehicles controlled by vehicle control agent)
blueprint = 'vehicle.tesla.model3'  # CARLA blueprint of vehicle (default = random)
speed_reduction_factor = 0          # factor to reduce the target speed of vehicles [%] (default = 0.0)
auto_pilot = true                   # let vehicle be controlled by auto-pilot (default = true)
use_traffic_manager = false         # use traffic manager for auto-pilot if true otherwise use vehicle control agent
                                    # (default = true)
quantity = 1                        # number of vehicles spawned from this spawner (default = 1)
spawn_time = 0.0                    # spawn time [s] of first vehicle spawning with this spawner (default = 0.0)
spawn_interval = 5.0                # time interval [s] between spawning of vehicles (default = 5.0)
ignore_lights_percentage = 100      # percentage of traffic lights the vehicle should ignore (default = 0.0)
ignore_walkers_percentage = 0       # percentage of pedestrians the vehicle should ignore (default = 0.0)


# general obstacle configurations
[obstacles]
resolution = 0.1                # resolution of border points [m]
ellipse_shape = true            # use generated ellipse-shaped obstacle borders (default = true)
max_obstacle_z_pos = 0.3        # ignore obstacles above z-position [m] (default = 0.3]

# definition of manually placed borderlines
[[obstacles.borders]]
start_point = [-53.9, 59.8]     # start point of borderline [x,y]
end_point = [-53.9, 29.8]       # end point of borderline [x,y]
```

In `config/scenarios` there are several more example scenarios.
A specific scenario can be executed by running:
 ```sh
   python run_simulation.py --scenario-config=<path-to-scenario-config-file>
   ```

## Acknowledgements

The implementation of the core SFM and the calculation of the forces is based on the implementations of
[yuxiang-gao](https://github.com/yuxiang-gao/PySocialForce) and [svenkreiss](https://github.com/svenkreiss/socialforce).

## References

<a id="1">[1]</a> Moussaïd, Mehdi, et al. "Experimental study of the behavioural mechanisms underlying self-organization
in human crowds." Proceedings of the Royal Society B: Biological Sciences 276.1668 (2009): 2755-2762.
<https://doi.org/10.1098/rspb.2009.0405>