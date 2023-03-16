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
1. Install CARLA according to its [documentation](https://carla.readthedocs.io/en/latest/start_quickstart/) 
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

### Scenario configuration

The simulation scenario can be defined and configured with a `scenario_config.toml` file.
In `config/scenarios` there are several example scenarios.

A specific scenario can be executed by running:
 ```sh
   python run_simulation.py --scenario-config=<path-to-scenario-config-file>
   ```

## Acknowledgements

The implementation of the core SFM and the calculation of the forces is based on the implementations of
[yuxiang-gao](https://github.com/yuxiang-gao/PySocialForce) and [svenkreiss](https://github.com/svenkreiss/socialforce).

## References

<a id="2">[1]</a> Moussaïd, Mehdi, et al. "Experimental study of the behavioural mechanisms underlying self-organization
in human crowds." Proceedings of the Royal Society B: Biological Sciences 276.1668 (2009): 2755-2762.
<https://doi.org/10.1098/rspb.2009.0405>