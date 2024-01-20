<center><font size=6>Codes and Interactive Interface of Saint Paul</font></center>

# Introduction

This code is for the algorithm we developed in the paper "Approximating the Performance of Three-State Emergency Service Systems." It includes an interactive tool using real EMS data to aid the operational decisions of EMS managers.

In this project, we provide the code for the algorithm discussed in the paper, which analyzes the three-state emergency service system using an exact method, our approximation method, and simulation. Additionally, an interactive interface is available, demonstrating how our algorithm and code can be applied in a real-life scenario in Saint Paul, Minnesota. Users can drag and allocate different units and receive immediate results. A demonstration video is also provided.



# Installation Instructions

## Requirements

- Algorithm Codes: 
  - Python is required.
  - Both MacOS and Windows have been tested and are confirmed to work.
- Interactive Tool:
  - Windows is required to run the interactive tool.

## Codes

- Algorithm Codes: 
  - Place `Three_States.py` and `block_prob.py` in the same folder.
  - In `Three_States.py`, a sample code of implementation is provided.
- Interactive Interface:
  - Download the zipped files from https://ufile.io/8dbe0ovg.
  - Run `map_hypercube.exe` to use the interactive tool.
    - Note that `map_hypercube.exe` should be placed in the same directory with directory `tracts` and `figures`



# Algorithm Codes

## Structure

- `block_prob.py`: Helper functions
- `Three_States.py`: Main classes and functions
  - `class Two_State_Hypercube()`: Hypercube model for two-state systems
  - `class Three_State_Hypercube()`: Hypercube model for three-state systems

## Sample Implementation

```python
# Initialize a three-state system with
# Type-1 calling rate 8, average servicing rate 25
# Type-2 calling rate 10, average servicing rate 20
system = Three_State_Hypercube({'Lambda_1': 8, 'Mu_1': 25, 'Lambda_2': 10, 'Mu_2': 20})

# Update the total number of units to N=7
# Assign N_1=2 units for type-1
# Assign N_2=3 units for type-2
# Set number of atoms K=30
system.Update_Parameters(N=7, N_1=2, N_2=3, K=30)

# Create two subsystems
system.Creat_Two_Subsystems()

# Initialize two subsystems with heterogeneous servicing:
# Center servicing rates around 25 and 20
system.sub1.Random_Mu(25, radius=0.8)
system.sub2.Random_Mu(20, radius=0.8)

# Generate random preference lists
system.sub1.Random_Pref(seed=1)
system.sub2.Random_Pref(seed=1)

# Generate random time matrices
system.sub1.Random_Time_Mat(t_min=1, t_max=10, seed=1)
system.sub2.Random_Time_Mat(t_min=1, t_max=10, seed=1)

# Allocate arriving rates randomly among all atoms
system.sub1.Random_Fraction(seed=1)
system.sub2.Random_Fraction(seed=1)

# Solve the system using the exact method
system.Solve_3state_Hypercube()
print(system.rho_hyper_1, system.rho_hyper_2, system.Get_MRT_3state())

# Solve the system using our approximation method
system.Reset_Alpha()
system.Linear_Alpha()
print(system.sub1.rho_approx, system.sub2.rho_approx, system.Get_MRT_Approx_3state())

# Solve the system using simulation
MRT_1, MRT_2, _, rho_sim_1, rho_sim_2, _ = system.Simulator_Mu_nj(type="vec")
print(rho_sim_1, rho_sim_2, MRT_1, MRT_2)
```



# Interactive Interface

## Icons

| Icon                                          | Action                                                       |
| --------------------------------------------- | ------------------------------------------------------------ |
|Joint unit| Drag from the sidebar to add a station.<br>Drag units on the map to change their location.<br>Right-click to delete from the map |
 | Fire unit                                     | Drag from the sidebar to add a station.<br>Drag units on the map to change their location.<br>Right-click to delete from the map |
| Medical unit                                  | Drag from the sidebar to add a station.<br>Drag units on the map to change their location.<br>Right-click to delete from the map |
 | Begin evaluation                              | Left-click to run                                            |
 | Allocate units<br>based on real configuration | Left-click to show the configuration                         |
 | Switch between<br>fire and EMS assessment     | Left-click to switch                                         |
 | Show or hide<br>station utilization           | Left-click to switch                                         |
 | Clear all units on the map                    | Left-click to clear                                          |

## Implementations

### Configure allocations

- Drag existing units on the map to their location.
- Drag from the right sidebar to add units:
  - Black circle: a joint unit.
  - Red square: an EMS unit.
  - Blue square: a fire unit.
- Right-click on units on the map to delete a unit.
- Clear all units.

### Assess system

- The utilization of every unit in both calls is shown around the icon.
- Mean response time for each district is displayed on the map:
  - The values, plotted in black in the middle of each district, show the Mean Response Time (MRT).
  - Districts are color-coded, with green denoting shorter response times and red denoting longer response times.
  - Total response times for both the fire and EMS systems are visualized at the bottom.
- Values are immediately updated after each drag or other configuration changes.



# Reference

- Approximating the Performance of Three-State Emergency Service Systems. 

## Contact

Hidden for double-blind paper
