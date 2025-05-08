# Main Setup Guide for Framework
This setup_guide will function as a walkthrough on how to install this framework on your machine using a virtual conda environment. The beginning of this tutorial will go through the basics on downloading the files and creating the environment. Once this first step is complete, this guide will dive into how to use the code properly. Below you will find badges for code coverage for different OS systems and current status of test cases.

[![codecov](https://codecov.io/gh/K-batonisashvili/PO-RRT-Pareto-Optimal-Path-Planning/graph/badge.svg?token=JSEBHATNLH)](https://codecov.io/gh/K-batonisashvili/PO-RRT-Pareto-Optimal-Path-Planning)

[![Run Tests](https://github.com/K-batonisashvili/PO-RRT-Pareto-Optimal-Path-Planning/actions/workflows/Tests.yml/badge.svg)](https://github.com/K-batonisashvili/PO-RRT-Pareto-Optimal-Path-Planning/actions/workflows/Tests.yml)

[![python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)

![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)

- `README.md`: Introduction to the Github repo, research, and algorithm.
- `setup_guide.md`: Main tutorial on how to install the repo (specifically through conda) and troubleshooting.
- `src folder`: Contains the main code for our PO-RRT* algorithm and supporting tools.
- `tests folder`: Contains the test functions to be used with Pytest. This ensures proper operation of our code.
- `imgs folder`: Contains graphics used in these guides. Does not contain algorithm-related material.
- `GenAIuse.txt`: Contains the statement describing how AI was used for this research.
- `pyproject.toml`: Contains all the requirements for getting the library setup.

This repository ensures that all professional code requirements learned from ME700 are met:
1) Cross-OS compatibility
2) Code coverage for all systems
3) Test-Driven-Development for stability, verification, and sanity checks
4) General research explanation on the main README with an additional setup_guide document
5) Robust installation via virtual environment or through raw zip file download
6) Industry standard in code organization
7) Branch and version control for alternate implementation methods
8) Robust documentation and annotation of tests, source files, setup, readme
9) Technical correctness ensured in full completion of all tests
10) Code optimization through hierarchical and modular class setup
11) Code optimization through alternate library imports specific to our PO_RRT_star algorithm

## Getting Started
When cloning this repo, you will have a general `src` (source) folder with all the primary files within it. The `tests` folder contains over 40 Pytest functions that are used in troubleshooting and verification of the code. `README.md`, `setup_guide.md`, and `pyproject.toml` will be in the base directory next to the `src` folder. The rest of the setup for this should be fairly straightforward.


We are going to create a new conda environment for this code for ease of use and to better keep track of all dependencies. Explicitly setting up a virtual environment ensures that future library installations will not be affected by our code today.

If you do not have conda installed on your computer, please visit the website below and follow all the instructions on setting up Miniconda. If you have a dedicated virtual environment already running or have a different preference, you may use that instead. Additionally, installing a virtual environment is not a hard requirement for this framework to run, it simply consolidates everything neatly into one location and has fewer errors when running. Alternatively, you may download the raw zip file from github and directly run the python file in the source folder.

Miniconda download: https://www.anaconda.com/docs/getting-started/miniconda/main

Once miniconda is set up on your computer, please follow the commands below to create your conda environment.

### Conda Environment Setup

Begin by setting up a conda or mamba environment:
```bash
conda create --name po-rrt-env python=3.12.9
```
Once the environment has been created, activate it:

```bash
conda activate po-rrt-env
```
Double check that python is version 3.12 in the environment:
```bash
python --version
```
Ensure that pip is using the most up to date version of setuptools:
```bash
pip install --upgrade pip setuptools wheel
```
If you have a folder from which you are working from, please navigate there now. Otherwise, create a new folder to clone this repo and then navigate into it:
```bash
mkdir PO-RRT-Repo
cd PO-RRT-Repo
```
We need to git clone the original repo and then navigate to the PO-RRT-Repo to install all the pre-requisites. In your terminal, please write the following:
```bash
git clone https://github.com/K-batonisashvili/PO-RRT-Pareto-Optimal-Path-Planning.git
```
Once you clone the directory, lets navigate to the appropriate sub-directory. Your path should look similar to this:
```bash
cd ./PO-RRT-Pareto-Optimal-Path-Planning
```
Finally, with this pip install command, we will download all required packages and dependencies into your conda environment:
```bash
pip install -e .
```
Thats it!

## Tutorial Deep Dive

In this part of the setup guide, we will examine how our code functions, the potential modifications, applications, edge cases, and more. 


### Code Structure

Lets take a quick look at how the 3 main source files communicate with each other. `PO_RRT_Star.py` is our primary python file where our algorithm is stored. This file calls on visualization.py and helper_functions.py which contain their relevant functionality, creating plots and mathematical calculations, respectively. Our tests* folder contains several Pytest cases used while developing our code. You may browse through these test cases, edit current ones, or add new ones. These tests call on all three source files depending on necessity and the nature of the test itself. This code structure may be seen in the high-level overview flowchart below.  

![High-Level-Overview](./imgs/High-Level-Overview-Code.drawio.png)

### Running the Code

Running our PO_RRT_Star code is fairly straightforward. Once all the initialization steps are complete, simply run `PO_RRT_Star.py` in your terminal or in your dedicated GUI environment. An input dialog box should come up which asks for requested failure probabilities for your path. This will force the algorithm to grow the tree until your desired paths are found. Alternatively, you may also set maximum number of iterations and return all possible Pareto-Dominant paths found by the algorithm. Variables currently have defaults set up where a generic start and goal node is placed in a 4x4 grid. We created 4 obstacles to simulate a simple environment with some challenges. You may modify any of these variables and to learn more about this, please refer to the PO_RRT_Star Modifications section below. 


### PO_RRT_Star Modifications

The main algorithm operates on a class structure and divides core components into their respective sections. Since this work is continually worked on, it is subject to future changes and some of the information on this guide might be outdated. We will keep everything up to date to the best of our ability when major changes happen.
We created the following divisions for clarity and organization:
- Tree Class
- Path Class
- Node Class
- Grid Class
- PO_RRT_Star Method
- Main Method

There are constant variables defined at the top of the file which may be changed based on preference. Please note, changing these constants will result in small to drastic differences in how the algorithm functions. It is important to set these variables to reasonable parameters. While the central PO_RRT_Star will retain its core functionality, the results will vary. For example, setting default step size to be greater than your plot size (X_MIN, X_MAX, Y_MIN, Y_MAX) will obviously yield incorrect results.

In the main class, we need to instantiate our start and goal nodes, create our environment and its potential obstacles, call our PO_RRT_Star function, and then call the metric plotting function. Failure probability values are chosen by the user to determine overall probability of failure per path (read: we want to find paths with a 10% probability of failure, 30%, 40%, etc). You may forego the input dialog box entirely and manually set up desired values as well.  

To create additional obstacles, we currently have templates to create circular and rectangular shapes with a dedicated probability of failure for each. Circular obstacles follow a gradient decline in terms of occupancy the further away you are from the obstacles. Rectangular obstacles have uniform occupancy across the entire area. You may add as many obstacles as you wish, and overlapping obstacles will prioritize the highest probability in the grid. 



### TROUBLESHOOTING: If you encounter "helper functions module not available" or "visualization module not available"

If your code is running into issues where errors display a message similar to "unable to find library", that might mean one of 2 things, please try both fixes.

1) Make sure your conda environment is activated in the terminal. Please do so by typing `conda activate po-rrt-env` in your terminal. Then run the command `conda list` which should include the po-rrt-env library.
2) When running in VScode directly, please make sure that your interpreter for the Jupyter Notebook is running the conda environment instance. In VScode, press CTRL SHIFT P and type in "interpreter" and go through the menu to select your conda env.


NOTE: If you chose to download the raw zip from github without installing the virtual library, you might need to shuffle your files around to make this work. We found it easiest to combine the 3 files into 1 central python file and running it directly.

### TROUBLESHOOTING: Potential Cluster Server or SCC errors

This code is set up in a way where an input dialog box comes up asking for the user to input several potential probabilities to seek when generating paths. Due to the nature of this dialog box and how computing clusters operate, visualization for this box might not function properly. To address this, please comment out all lines for:
`failure_prob_values = simpledialog.askstring("Input", "Enter the failure probabilities (comma-separated):")` (or this would be approximately lines 500 - 505 in the main PO_RRT_Star.py file)
and replace failure_prob_values with your own chosen number.