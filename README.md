# ADPRO_Project_Group_02


## Description

This project encompasses a suite of tools designed for the analysis and visualization of flight data. At its core, the `FlightDataAnalyzer` class provides comprehensive functionalities for managing and interpreting flight information, leveraging external APIs for enriched data acquisition. Features include distance calculations between airports, visualization of flight paths on global maps, and fetching detailed airport specifications for enhanced insights.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

1. Clone the GitLab repository to your local machine:
```
git clone git@gitlab.com:https://gitlab.com/group_adpro_02/ADPRO_Project.git
```
2. Navigate to the project directory:

```
cd ADPRO_Project
```
3. Set up your virtual environment with conda:
```
conda env create -f group_02_env
```
4. Run tests to make sure everything is running accordingly.
First, navigate to the `FlightAnalyser` directory:
```
cd FlightAnalyser
``` 
Then, run the following command:
```
pytest function_test_distance.py
```


## Usage

- **Distance Calculation:** Utilize `calculate_distance` to find the distance between two airports.
- **Flight Path Visualization:** Generate interactive maps showing flight routes using data processed by the `FlightDataAnalyzer` class.
- **Airport Specifications:** Fetch and display detailed information about airports, aiding in a deeper understanding of flight data.


## Authors

Ethan Liegon <a href="mailto:59527@novasbe.pt">59527@novasbe.pt</a><br>
Martin Haag <a href="mailto:61745@novasbe.pt">61745@novasbe.pt</a><br>
Jona Weishaupt <a href="mailto:61374@novasbe.pt">61374@novasbe.pt</a><br>
Martim Alves Ernesto Esteves <a href="mailto:46953@novasbe.pt">46953@novasbe.pt</a>


## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE. See the [LICENSE](LICENSE) file for details.
