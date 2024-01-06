<center><font size=6>Codes and Interactive Interface of Saint Paul</font></center>

# Introduction

## Abstract

This is codes for "Approximating the Performance of Three-State Emergency Service Systems" and interactive interface in Saint Paul, Minnesota. 

In this project, we provide the algorithm codes in this paper, which is used to analyze the stochastic system with exact method, approximation method, and simulation. Also, an interactive interface is provided to show how our algorithm and codes can be used in a real-life case in Saint Paul, Minnesota, where you can drag and allocation different stations and get the metrics immediately. 

## Contact Information

- Professor Cheng Hua
  - Institute: Antai College of Economics and Management, Shanghai Jiao Tong University
  - E-mail: cheng.hua@sjtu.edu.cn
- Tianyi Ma
  - Institute: University of Michigan-Shanghai Jiao Tong University Joint Institute
  - E-mail: sean_ma@sjtu.edu.cn




# Installation Instructions

## Requirement

- Algorithm codes: 
  - Python is required
  - Both MacOS and Windows are tested to work
- Interactive interface:
  - Windows is required to run `map_hypercube.exe`
  - Both MacOS and Windows can view the film that indicates how our interface works
  - Python is not required here.

## Procedure

- Algorithm codes: 
  - Compile `Three_States.py` and `block_prob.py` together
  - In `Three_States.py`, a sample codes of implementation is shown. You can also implement by yourself.
- Interactive interface:
  - Download `map_hypercube.exe` from xxxx
  - `map_hypercube.exe` should be placed in the same directory with directory `tracts` and `figures`
  - Double click `map_hypercube.exe` and you can see the interface
  



# Algorithm Codes

## Structure

- `block_prob.py`: Help functions
- `Three_States.py`: main classes and functions
  - `class Two_State_Hypercube()`: Hypercube model that works for "busy" and "free" two states
  - `class Three_State_Hypercube()`: Hypercube model that works for "busy in 1", "busy in 2", and "free" three states

## Sample Implementation

```python
# Initialize a three state system with
# Type-1 calling rate 8, average servicing rate 25
# Type-2 calling rate 10, average servicing rate 20
system = Three_State_Hypercube({'Lambda_1': 8, 'Mu_1': 25, 'Lambda_2': 10, 'Mu_2': 20})

# Update the number of units totally N=7
# Separate units for type-1 as N_1=2
# Separate units for type-2 as N_2=3
# Number of atoms K=30
system.Update_Parameters(N=7, N_1=2, N_2=3, K=30)

# Create two subsystems to approximate
system.Creat_Two_Subsystems()

# Initialize two subsystems with
# Heterogeneous servicing among unit centered at 25 / 20
system.sub1.Random_Mu(25, radius=0.8)
system.sub2.Random_Mu(20, radius=0.8)

# Randomly generate preference list
system.sub1.Random_Pref(seed=1)
system.sub2.Random_Pref(seed=1)

# Randomly generate time matrix
system.sub1.Random_Time_Mat(t_min=1, t_max=10, seed=1)
system.sub2.Random_Time_Mat(t_min=1, t_max=10, seed=1)

# Randomly allocate arriving rate among all atoms
system.sub1.Random_Fraction(seed=1)
system.sub2.Random_Fraction(seed=1)

# Solve the system with exact method
system.Solve_3state_Hypercube()
print(system.rho_hyper_1, system.rho_hyper_2, system.Get_MRT_3state())

# Solve the system with approximation method
system.Reset_Alpha()
system.Linear_Alpha()
print(system.sub1.rho_approx, system.sub2.rho_approx, system.Get_MRT_Approx_3state())
```



# Interactive Interface

## Icons

| Icon                                                         | Meaning                                    | Action                                                       |
| ------------------------------------------------------------ | ------------------------------------------ | ------------------------------------------------------------ |
| <img src="./figures/joint.png" alt="joint" style="zoom:110%;" /> | Joint station                              | Drag from side bar to add station <br>Drag units on the map to change location<br> Right click to delete from map |
| <img src="./figures/fire.png" alt="fire" style="zoom:120%;" /> | Fire station                               | Drag from side bar to add station <br>Drag units on the map to change location<br>Right click to delete from map |
| <img src="./figures/med.png" alt="med" style="zoom:120%;" /> | Medical station                            | Drag from side bar to add station<br>Drag units on the map to change location<br/>Right click to delete from map |
| <img src="./figures/start.png" alt="start" style="zoom:8%;" /> | Start evaluation<br>for real configuration | Left click to run                                            |
| <img src="./figures/stpaul.jpg" alt="stpaul" style="zoom:9%;" /> | Allocate stations<br>based on real data    | Left click to configure                                      |
| <img src="./figures/switch.png" alt="switch" style="zoom:15%;" /> | Switch between<br>fire and EMS assessment  | Left click to switch                                         |
| <img src="./figures/uti.png" alt="uti" style="zoom:90%;" />  | Show or hide<br>station utilization        | Left click to switch                                         |
| <img src="./figures/clear-icon.png" alt="clear-icon" style="zoom:20%;" /> | Clear all units on map                     | Left click to Clear                                          |

## Implementations

### Configure allocations

- Simulate for real configuration in Saint Paul
- Drag existing icons on the map to change location
- Drag from right sidebar to add stations
  - Black circle: a joint unit,
  - Red square: an EMS unit
  - Blue square: a fire unit
- Right click on icons on map to delete a unit
- Clear all stations

### Assess system

- Utilization of every unit in both calls are shown around the icon
- Mean response time for each district is displayed on map
  - The values plotted in black in the middle of each district show the MRT
  - Districts are color-coded where green denotes a shorter response time and red denotes a longer response time
  - Total response time are visualized for both fire system and EMS system in the bottom
- Values are immediately changed after each drag or other configurations



# Reference

Paper:

Saint Paul, Minnesota

