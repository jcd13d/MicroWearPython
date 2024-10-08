import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import csv
import numpy as np
from itertools import combinations
import argparse
import os
import datetime  # Import datetime for timestamp

class MicroWear:
    def __init__(self, image_path, working_area_size=200):
        self.image_path = image_path  # Store image path for filename extraction
        self.image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]
        self.area = min(self.height, self.width) / 2
        self.working_area_size = working_area_size
        self.working_area = None
        self.traces = []
        self.scale_factor = None  # Initialize scale_factor

    def set_scale(self):
        fig, ax = plt.subplots()
        ax.imshow(self.image_rgb)
        ax.set_title("Click two points to set scale")
        
        points = plt.ginput(2)
        
        pixel_distance = np.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
        
        real_distance = float(input("Enter the real-world distance between the two points (in μm): "))
        
        self.scale_factor = real_distance / pixel_distance
        print(f"Scale set: 1 pixel = {self.scale_factor:.4f} μm")
        
        plt.close()

    def select_working_area(self):
        if self.scale_factor is None:
            print("Please set the scale first.")
            return

        # Convert working_area_size microns to pixels
        box_size_pixels = int(self.working_area_size / self.scale_factor)

        fig, ax = plt.subplots()
        ax.imshow(self.image_rgb)
        
        rect = Rectangle((0, 0), box_size_pixels, box_size_pixels, 
                         linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        def on_click(event):
            if event.button == 1:  # Left click
                x, y = int(event.xdata), int(event.ydata)
                rect.set_xy((x, y))
                fig.canvas.draw()

        def on_key(event):
            if event.key == 'enter':
                x, y = rect.get_xy()
                self.working_area = (int(x), int(y), 
                                     int(x + box_size_pixels), 
                                     int(y + box_size_pixels))
                plt.close()
                print(f"Working area set: {self.working_area}")

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.title("Click to position the 200x200 micron box. Press Enter to confirm.")
        plt.show()

    def sample_traces(self):
        if self.working_area is None:
            print("Please select a working area first.")
            return

        x1, y1, x2, y2 = self.working_area
        buffer = int(0.2 * (x2 - x1))  # 20% buffer
        working_image = self.image_rgb[max(0, y1-buffer):min(self.height, y2+buffer), max(0, x1-buffer):min(self.width, x2+buffer)]

        fig, (ax, status_ax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 12))
        ax.imshow(working_image)
        rect = Rectangle((buffer, buffer), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        status_ax.axis('off')

        current_sample = []
        all_samples = []
        sample_counter = 1
        selection_counter = 0
        quit_flag = False

        status_text = status_ax.text(0.5, 0.5, "", ha='center', va='center', fontsize=12)
        help_box = ax.text(0.01, 0.99, "", transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        def update_status():
            status_text.set_text(f"Sample: {sample_counter}\nSelection: {selection_counter}/4\nTotal Samples: {len(all_samples)}")
            fig.canvas.draw_idle()

        def add_help_box():
            help_text = ("Controls:\n"
                         "n: Next sample\n"
                         "u: Undo last point\n"
                         "c: Cancel current sample\n"
                         "q: Quit sampling\n"
                         "h: Toggle this help")
            help_box.set_text(help_text)
            help_box.set_visible(True)
            fig.canvas.draw_idle()

        def toggle_help_box():
            help_box.set_visible(not help_box.get_visible())
            fig.canvas.draw_idle()

        def draw_sample(sample, color='r'):
            line, = ax.plot([sample[0][0], sample[1][0]], [sample[0][1], sample[1][1]], color)
            return line

        add_help_box()
        update_status()

        sample_lines = []

        def on_key(event):
            nonlocal current_sample, all_samples, sample_counter, selection_counter, quit_flag, sample_lines, help_box
            if event.key == 'n':  # Next sample
                if len(current_sample) == 4:
                    all_samples.append(current_sample)
                    current_sample = []
                    sample_counter += 1
                    selection_counter = 0
                    sample_lines = []
                    ax.clear()
                    ax.imshow(working_image)
                    rect = Rectangle((buffer, buffer), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    for sample in all_samples:
                        draw_sample(sample[:2], 'r')
                        draw_sample(sample[2:], 'b')
                    # Re-add the help box after clearing the axes
                    help_box.remove()
                    help_box = ax.text(0.01, 0.99, "", transform=ax.transAxes, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                    add_help_box()
                else:
                    print("Please complete the current trace (4 points) before moving to the next.")
            elif event.key == 'u':  # Undo last point
                if current_sample:
                    current_sample.pop()
                    selection_counter = max(0, selection_counter - 1)
                    if sample_lines:
                        sample_lines.pop().remove()
                elif all_samples:
                    current_sample = all_samples.pop()
                    sample_counter = max(1, sample_counter - 1)
                    selection_counter = len(current_sample)
                    ax.clear()
                    ax.imshow(working_image)
                    for sample in all_samples:
                        draw_sample(sample[:2], 'r')
                        draw_sample(sample[2:], 'b')
                    if len(current_sample) >= 2:
                        sample_lines = [draw_sample(current_sample[:2], 'r')]
                    if len(current_sample) == 4:
                        sample_lines.append(draw_sample(current_sample[2:], 'b'))
                    # Re-add the help box after clearing the axes
                    help_box.remove()
                    help_box = ax.text(0.01, 0.99, "", transform=ax.transAxes, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                    add_help_box()
                else:
                    print("Nothing to undo.")
            elif event.key == 'c':  # Cancel current sample
                current_sample = []
                selection_counter = 0
                for line in sample_lines:
                    line.remove()
                sample_lines = []
            elif event.key == 'q':  # Quit sampling
                quit_flag = True
            elif event.key == 'h':  # Toggle help box
                toggle_help_box()
            update_status()
            fig.canvas.draw_idle()

        def on_click(event):
            nonlocal current_sample, selection_counter, sample_lines
            if event.inaxes == ax:
                if len(current_sample) < 4:
                    current_sample.append((event.xdata, event.ydata))
                    selection_counter += 1

                    if len(current_sample) == 2:
                        sample_lines.append(draw_sample(current_sample, 'r'))
                    elif len(current_sample) == 4:
                        sample_lines.append(draw_sample(current_sample[2:], 'b'))
                        length = np.sqrt((current_sample[1][0] - current_sample[0][0])**2 + 
                                         (current_sample[1][1] - current_sample[0][1])**2)
                        width = np.sqrt((current_sample[3][0] - current_sample[2][0])**2 + 
                                        (current_sample[3][1] - current_sample[2][1])**2)
                        if width > length:
                            length, width = width, length
                        self.traces.append({
                            'length': length * self.scale_factor,
                            'width': width * self.scale_factor,
                            'start': current_sample[0],
                            'end': current_sample[1]
                        })

                    update_status()
                    fig.canvas.draw_idle()
                else:
                    print("You've already selected 4 points for this sample. Press 'n' to start a new sample or 'u' to undo the last point.")
                    status_text.set_text("Max 4 points reached.\nPress 'n' for new sample\nor 'u' to undo.")
                    fig.canvas.draw_idle()

        def on_close(event):
            nonlocal quit_flag
            quit_flag = True

        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('close_event', on_close)

        while not quit_flag:
            plt.pause(0.1)

        plt.close(fig)
        print("Sampling completed.")

    def classify_traces(self):
        for trace in self.traces:
            if trace['length'] / trace['width'] > 4:
                trace['type'] = 'Scratch'
                trace['subtype'] = 'Fine' if trace['width'] <= 3 else 'Coarse'
            else:
                trace['type'] = 'Pit'
                trace['subtype'] = 'Small' if trace['length'] <= 8 else 'Large'

    def calculate_area_mm2(self):
        if self.working_area and self.scale_factor:
            x1, y1, x2, y2 = self.working_area
            width_pixels = x2 - x1
            height_pixels = y2 - y1
            width_mm = width_pixels * self.scale_factor
            height_mm = height_pixels * self.scale_factor
            return (width_mm * height_mm) / 1e6
        return 0

    def detect_parallel_and_crossing_scratches(self):
        scratches = [trace for trace in self.traces if trace['type'] == 'Scratch']
        results = []

        threshold = (self.area * 2) * 2

        for i, j in combinations(range(len(scratches)), 2):
            scratch1 = scratches[i]
            scratch2 = scratches[j]

            # Calculate intersection point
            intersection = self.calculate_intersection(scratch1, scratch2)

            # Determine if scratches are crossing
            is_crossing = False
            if intersection:
                x, y = intersection
                x1_range = [min(scratch1['start'][0], scratch1['end'][0]), max(scratch1['start'][0], scratch1['end'][0])]
                y1_range = [min(scratch1['start'][1], scratch1['end'][1]), max(scratch1['start'][1], scratch1['end'][1])]
                x2_range = [min(scratch2['start'][0], scratch2['end'][0]), max(scratch2['start'][0], scratch2['end'][0])] 
                y2_range = [min(scratch2['start'][1], scratch2['end'][1]), max(scratch2['start'][1], scratch2['end'][1])]
                
                is_crossing = (x1_range[0] < x < x1_range[1] and
                               x2_range[0] < x < x2_range[1] and
                               y1_range[0] < y < y1_range[1] and
                               y2_range[0] < y < y2_range[1])

            # Determine if scratches are parallel
            is_parallel = False
            if intersection:
                dist1 = self.point_to_line_distance(intersection, scratch1['start'], scratch1['end'])
                dist2 = self.point_to_line_distance(intersection, scratch2['start'], scratch2['end'])
                min_dist = min(dist1, dist2)
                is_parallel = not (min_dist < threshold)  # Note the change here

            # Calculate angle between scratches
            angle = self.calculate_angle(scratch1, scratch2)

            results.append({
                'scratch1': i,
                'scratch2': j,
                'is_crossing': is_crossing,
                'is_parallel': is_parallel,
                'angle': angle
            })

        return results

    def calculate_intersection(self, scratch1, scratch2):
        x1, y1 = scratch1['start']
        x2, y2 = scratch1['end']
        x3, y3 = scratch2['start']
        x4, y4 = scratch2['end']

        det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(det) < 1e-8:  # Lines are parallel or coincident
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / det
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return (x, y)

    def point_to_line_distance(self, point, line_start, line_end):
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Calculate distances to start and end points
        dist_start = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        dist_end = np.sqrt((x0 - x2)**2 + (y0 - y2)**2)

        # Return the minimum distance
        return min(dist_start, dist_end)

    def calculate_angle(self, scratch1, scratch2):
        v1 = np.array(scratch1['end']) - np.array(scratch1['start'])
        v2 = np.array(scratch2['end']) - np.array(scratch2['start'])
        dot_product = np.dot(v1, v2)
        magnitudes = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        # Handle potential floating-point errors
        cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle

    def generate_summary(self, output_file=None):
        summary = {
            'Pit': {'total': 0, 'small': 0, 'large': 0, 'lengths': [], 'widths': []},
            'Scratch': {'total': 0, 'fine': 0, 'coarse': 0, 'lengths': [], 'widths': []}
        }
        
        for trace in self.traces:
            summary[trace['type']]['total'] += 1
            summary[trace['type']][trace['subtype'].lower()] += 1
            summary[trace['type']]['lengths'].append(trace['length'])
            summary[trace['type']]['widths'].append(trace['width'])
        
        area_mm2 = self.calculate_area_mm2()
        scratch_results = self.detect_parallel_and_crossing_scratches()
        
        parallel_pairs = sum(1 for result in scratch_results if result['is_parallel'])
        crossing_pairs = sum(1 for result in scratch_results if result['is_crossing'])
        
        stats = {
            'N.pits': summary['Pit']['total'],
            'N.sp': summary['Pit']['small'],
            'N.lp': summary['Pit']['large'],
            '%p': (summary['Pit']['total'] / (summary['Pit']['total'] + summary['Scratch']['total'])) * 100 if (summary['Pit']['total'] + summary['Scratch']['total']) > 0 else 0,
            'P': summary['Pit']['total'] / area_mm2 if area_mm2 > 0 else 0,
            'N.scratches': summary['Scratch']['total'],
            'N.fs': summary['Scratch']['fine'],
            'N.cs': summary['Scratch']['coarse'],
            'S': summary['Scratch']['total'] / area_mm2 if area_mm2 > 0 else 0,
            'N.Ps': parallel_pairs,
            'N.Xs': crossing_pairs,
            '%Ps': (parallel_pairs / summary['Scratch']['total']) * 100 if summary['Scratch']['total'] > 0 else 0,
            '%Xs': (crossing_pairs / summary['Scratch']['total']) * 100 if summary['Scratch']['total'] > 0 else 0
        }
        
        # Calculate means and standard deviations
        for feature in ['Pit', 'Scratch']:
            for dimension in ['length', 'width']:
                for subtype in ['small', 'large'] if feature == 'Pit' else ['fine', 'coarse']:
                    data = [trace[dimension] for trace in self.traces if trace['type'] == feature and trace['subtype'].lower() == subtype]
                    stats[f'{feature.lower()}_{subtype}_mean_{dimension}'] = np.mean(data) if data else 0
                    stats[f'{feature.lower()}_{subtype}_sd_{dimension}'] = np.std(data) if data else 0
                
                all_data = summary[feature][dimension + 's']
                stats[f'{feature.lower()}_mean_{dimension}'] = np.mean(all_data) if all_data else 0
                stats[f'{feature.lower()}_sd_{dimension}'] = np.std(all_data) if all_data else 0
        
        # Generate output filename if not provided
        if output_file is None:
            current_time = datetime.datetime.now()
            time_str = current_time.strftime('%Y%m%d%H%M%S')
            image_filename = os.path.splitext(os.path.basename(self.image_path))[0]
            # **Modified Filename Order Here**
            output_file = f"{image_filename}_summary_{time_str}.csv"
        
        # Write to CSV
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['', 'N.pits', 'N.sp', 'N.lp', '%p', 'P', 'N.scratches', 'N.fs', 'N.cs', 'S', 'N.Ps', 'N.Xs', '%Ps', '%Xs'])
            writer.writerow(['Count', stats['N.pits'], stats['N.sp'], stats['N.lp'], f"{stats['%p']:.1f}", 
                             f"{stats['P']:.2f}", stats['N.scratches'], stats['N.fs'], stats['N.cs'], f"{stats['S']:.2f}", 
                             stats['N.Ps'], stats['N.Xs'], f"{stats['%Ps']:.1f}", f"{stats['%Xs']:.1f}"])
            
            writer.writerow(['Mean_length', f"{stats['pit_mean_length']:.2f}", f"{stats['pit_small_mean_length']:.2f}", f"{stats['pit_large_mean_length']:.2f}", 
                             '/', '/', f"{stats['scratch_mean_length']:.2f}", f"{stats['scratch_fine_mean_length']:.2f}", 
                             f"{stats['scratch_coarse_mean_length']:.2f}", '/', '/', '/', '/', '/'])
            writer.writerow(['Sd_length', f"{stats['pit_sd_length']:.2f}", f"{stats['pit_small_sd_length']:.2f}", f"{stats['pit_large_sd_length']:.2f}", 
                             '/', '/', f"{stats['scratch_sd_length']:.2f}", f"{stats['scratch_fine_sd_length']:.2f}", 
                             f"{stats['scratch_coarse_sd_length']:.2f}", '/', '/', '/', '/', '/'])
            writer.writerow(['Mean_width', f"{stats['pit_mean_width']:.2f}", f"{stats['pit_small_mean_width']:.2f}", f"{stats['pit_large_mean_width']:.2f}", 
                             '/', '/', f"{stats['scratch_mean_width']:.2f}", f"{stats['scratch_fine_mean_width']:.2f}", 
                             f"{stats['scratch_coarse_mean_width']:.2f}", '/', '/', '/', '/', '/'])
            writer.writerow(['Sd_width', f"{stats['pit_sd_width']:.2f}", f"{stats['pit_small_sd_width']:.2f}", f"{stats['pit_large_sd_width']:.2f}", 
                             '/', '/', f"{stats['scratch_sd_width']:.2f}", f"{stats['scratch_fine_sd_width']:.2f}", 
                             f"{stats['scratch_coarse_sd_width']:.2f}", '/', '/', '/', '/', '/'])
        
        print(f"Summary statistics have been saved to {output_file}")
        
        # Print summary to console
        print("\nSummary Statistics:")
        print(f"N.pits: {stats['N.pits']}, N.sp: {stats['N.sp']}, N.lp: {stats['N.lp']}, %p: {stats['%p']:.1f}, P: {stats['P']:.2f}")
        print(f"N.scratches: {stats['N.scratches']}, N.fs: {stats['N.fs']}, N.cs: {stats['N.cs']}, S: {stats['S']:.2f}")
        print(f"N.Ps: {stats['N.Ps']}, N.Xs: {stats['N.Xs']}, %Ps: {stats['%Ps']:.1f}, %Xs: {stats['%Xs']:.1f}")
        print(f"Pit Mean Length: {stats['pit_mean_length']:.2f} ± {stats['pit_sd_length']:.2f}")
        print(f"Pit Mean Width: {stats['pit_mean_width']:.2f} ± {stats['pit_sd_width']:.2f}")
        print(f"Scratch Mean Length: {stats['scratch_mean_length']:.2f} ± {stats['scratch_sd_length']:.2f}")
        print(f"Scratch Mean Width: {stats['scratch_mean_width']:.2f} ± {stats['scratch_sd_width']:.2f}")

    def visualize_classified_traces(self):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Display the original image
        ax.imshow(self.image_rgb)

        # Draw the working area
        x1, y1, x2, y2 = self.working_area
        rect = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='yellow', linewidth=2)
        ax.add_patch(rect)

        # Calculate buffer
        buffer = int(0.2 * self.working_area_size / self.scale_factor)  # 20% buffer, dynamic based on working_area_size
        
        for i, trace in enumerate(self.traces):
            # Translate coordinates from the working area (including buffer) to the full image
            start_x = trace['start'][0] - buffer + self.working_area[0]
            start_y = trace['start'][1] - buffer + self.working_area[1]
            end_x = trace['end'][0] - buffer + self.working_area[0]
            end_y = trace['end'][1] - buffer + self.working_area[1]

            if trace['type'] == 'Pit':
                if trace['subtype'] == 'Small':
                    ax.plot([start_x, end_x], [start_y, end_y], 'ro-', markersize=5, linewidth=1)
                else:  # Large Pit
                    ax.plot([start_x, end_x], [start_y, end_y], 'ro-', markersize=8, linewidth=1)
            else:  # Scratch
                if trace['subtype'] == 'Fine':
                    ax.plot([start_x, end_x], [start_y, end_y], 'b-', linewidth=1)
                else:  # Coarse Scratch
                    ax.plot([start_x, end_x], [start_y, end_y], 'b-', linewidth=2)

            # Add label
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            ax.text(mid_x, mid_y, str(i+1), color='yellow', fontsize=8, 
                    ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5))

        # Draw the buffer zone
        buffer_rect = Rectangle((x1-buffer, y1-buffer), x2-x1+2*buffer, y2-y1+2*buffer, 
                                fill=False, edgecolor='green', linestyle='--', linewidth=1)
        ax.add_patch(buffer_rect)

        ax.set_title('Classified Microwear Traces')
        plt.tight_layout()
        plt.show()


    def save_traces_to_csv(self, output_file=None):
        if output_file is None:
            current_time = datetime.datetime.now()
            time_str = current_time.strftime('%Y%m%d%H%M%S')
            image_filename = os.path.splitext(os.path.basename(self.image_path))[0]
            # **Modified Filename Order Here**
            output_file = f"{image_filename}_traces_{time_str}.csv"

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Save scale factor and working area at the top of the CSV file
            writer.writerow(['Scale Factor', self.scale_factor])
            writer.writerow(['Working Area', self.working_area])
            # Proceed with writing the traces as before
            fieldnames = ['Trace Number', 'Type', 'Subtype', 'Length (μm)', 'Width (μm)',
                          'Start X (px)', 'Start Y (px)', 'End X (px)', 'End Y (px)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for idx, trace in enumerate(self.traces):
                writer.writerow({
                    'Trace Number': idx + 1,
                    'Type': trace.get('type', ''),
                    'Subtype': trace.get('subtype', ''),
                    'Length (μm)': trace['length'],
                    'Width (μm)': trace['width'],
                    'Start X (px)': trace['start'][0],
                    'Start Y (px)': trace['start'][1],
                    'End X (px)': trace['end'][0],
                    'End Y (px)': trace['end'][1]
                })

        print(f"Traces have been saved to {output_file}")

    def load_traces_from_csv(self, input_file):
        if not os.path.isfile(input_file):
            print(f"File {input_file} does not exist.")
            return

        with open(input_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            # Read scale factor and working area
            scale_row = next(reader)
            working_area_row = next(reader)
            self.scale_factor = float(scale_row[1])
            # Convert working area string back to tuple of integers
            self.working_area = tuple(map(int, working_area_row[1].strip('()').split(',')))
            # Proceed to read the traces
            reader = csv.DictReader(csvfile)
            self.traces = []
            for row in reader:
                trace = {
                    'type': row.get('Type', ''),
                    'subtype': row.get('Subtype', ''),
                    'length': float(row['Length (μm)']),
                    'width': float(row['Width (μm)']),
                    'start': (float(row['Start X (px)']), float(row['Start Y (px)'])),
                    'end': (float(row['End X (px)']), float(row['End Y (px)']))
                }
                self.traces.append(trace)

        print(f"Traces have been loaded from {input_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroWear Analysis Tool")
    IMAGE_PATH = 'images/paper_img_A.png'  # Replace with your image path
    parser.add_argument('--image_path', type=str, default=IMAGE_PATH, help='Path to the image file')
    parser.add_argument('--working_area_size', type=int, default=200, help='Size of the working area in microns (default: 200)')
    parser.add_argument('--trace_file', type=str, help='Path to the trace file to load')
    args = parser.parse_args()

    micro_wear = MicroWear(args.image_path, args.working_area_size)

    if args.trace_file:
        # Load traces from the provided trace file
        micro_wear.load_traces_from_csv(args.trace_file)
    else:
        # Proceed with interactive scale setting and sampling
        micro_wear.set_scale()
        micro_wear.select_working_area()
        micro_wear.sample_traces()
        # Save the measurements to a CSV file
        micro_wear.save_traces_to_csv()

    # Classify traces (ensure they are classified whether loaded or sampled)
    micro_wear.classify_traces()

    # Visualize the classified traces
    micro_wear.visualize_classified_traces()

    # Generate and save the summary statistics
    if not args.trace_file:
        micro_wear.generate_summary()
