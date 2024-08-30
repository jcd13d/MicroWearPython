# MicroWear Analysis Tool

## Description
The MicroWear Analysis Tool is a Python-based program designed for analyzing microwear patterns on tooth surfaces or similar materials. It allows researchers to manually sample and classify wear traces (pits and scratches) from microscope images, calculate various statistics, and visualize the results.

## Features
- Image loading and scale setting
- Interactive selection of working area
- Manual sampling of wear traces
- Automatic classification of traces into pits (small/large) and scratches (fine/coarse)
- Detection of parallel and crossing scratches
- Generation of summary statistics
- Visualization of classified traces
- CSV output of analysis results

## Requirements
- Anaconda or Miniconda

## Installation
1. Ensure you have Anaconda or Miniconda installed on your system.
2. Clone this repository:
   ```
   git clone https://github.com/your-username/microwear-analysis-tool.git
   cd microwear-analysis-tool
   ```
3. Create a new conda environment using the provided environment.yml file:
   ```
   conda env create -f environment.yml
   ```
4. Activate the new environment:
   ```
   conda activate jewels
   ```
   (feel free to create your own envi if you want)

## Usage
1. Run the script from the command line:
   ```
   python microwear_analysis.py --image_path <path_to_image> --working_area_size <size_in_microns>
   ```
   - `--image_path`: Path to the microscope image (default: 'images/paper_img_A.png')
   - `--working_area_size`: Size of the working area in microns (default: 200)

2. Follow the on-screen instructions to:
   - Set the scale by clicking two points and entering the real-world distance
   - Select the working area by positioning the box and pressing Enter
   - Sample traces by clicking to mark start and end points (4 points per trace)
   - Use keyboard controls during sampling:
     - 'n': Next sample
     - 'u': Undo last point
     - 'c': Cancel current sample
     - 'q': Quit sampling
     - 'h': Toggle help information

3. After sampling, the program will:
   - Classify the traces
   - Visualize the classified traces on the original image
   - Generate and display summary statistics
   - Save the summary statistics to a CSV file named 'microwear_summary.csv'

## Output
- A visualization of the classified traces overlaid on the original image
- A CSV file ('microwear_summary.csv') containing summary statistics, including:
  - Counts of different trace types
  - Percentages and densities of pits and scratches
  - Mean lengths and widths of different trace types
  - Standard deviations of lengths and widths
  - Counts and percentages of parallel and crossing scratches

## Notes
- The working area is set to 200x200 microns by default but can be adjusted using the `--working_area_size` argument.
- The program uses a 20% buffer around the working area for trace selection to allow for more flexibility in sampling.
- Traces are classified based on their length-to-width ratio and absolute size.
- The visualization includes numbered labels for each trace for easy reference.

## Acknowledgments
https://github.com/MicroWeaR/MicroWeaR but in python, all credit to original authors
