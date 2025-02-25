import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Patch
import os
import fitz
import json
from collections import defaultdict
from .rwc_jit import rwc
from .pads import pads_python
from .utils import get_graph
from matplotlib.lines import Line2D
from . import run_exp
from .opinion_dynamics import opinion_dynamics_reweight


def ec_ecc(source_path, crop_area):
    pdf_files = ['EC.pdf', 'ECC.pdf']

    # Create a new PDF document for merged result
    merged_pdf = fitz.open()

    # Get dimensions of first page after cropping
    doc = fitz.open(os.path.join(source_path, pdf_files[0]))
    page = doc[0]
    rect = fitz.Rect(*crop_area)
    width = rect.width
    height = rect.height
    doc.close()

    # Create new page with double height for merged PDF
    merged_pdf.new_page(width=width, height=height * 2)
    merged_page = merged_pdf[0]

    # Process each PDF
    for i, pdf_path in enumerate(pdf_files):
        # Open source PDF
        doc = fitz.open(os.path.join(source_path, pdf_path))
        page = doc[0]

        # Save individual cropped PDF
        new_doc = fitz.open()
        new_doc.new_page(width=width, height=height)
        new_page = new_doc[0]

        # Copy cropped content to new individual PDF
        new_page.show_pdf_page(
            fitz.Rect(0, 0, width, height),  # target rectangle
            doc,                             # source document
            0,                               # source page number
            clip=rect                        # crop area
        )

        # Save the individual cropped PDF
        cropped_filename = f'Cropped_{pdf_path}'
        new_doc.save(os.path.join(source_path, cropped_filename))
        new_doc.close()

        # Add to merged PDF
        y_offset = i * height
        target_rect = fitz.Rect(0, y_offset, width, y_offset + height)

        merged_page.show_pdf_page(
            target_rect,            # target rectangle
            doc,                    # source document
            0,                      # source page number
            clip=rect               # crop area
        )

        doc.close()

    # Save the merged PDF
    merged_pdf.save(os.path.join(source_path, 'Merged.pdf'))
    merged_pdf.close()


def joint_distribution(G, pos_value=1, neg_value=-1, save_path=None):
    # Set global font sizes
    sns.set_context("notebook", font_scale=1.5)  # Adjust font_scale as needed
    plt.rcParams.update({
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14,
        'font.size': 16,
    })

    method_names = {'maxflow_cpp_udsp': 'MaxFlow-U', 'maxflow_cpp_wdsp': 'MaxFlow-W', 'node2vec_gin': 'GIN', 'pads_cpp':
        'PADS'}
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # Increased figsize for better readability

    for ax, method_name in zip(axs, method_names.keys()):
        pos_corr = []
        pos_polarities = []
        neg_corr = []
        neg_polarities = []

        # Separate nodes based on their values
        for node in G.nodes:
            node_value = G.nodes[node].get(method_name, None)
            if node_value == pos_value or node_value == neg_value:
                neighbors = [neighbor for neighbor in G.neighbors(node) if G.nodes[neighbor].get(method_name, None) == node_value]
                if neighbors:
                    avg = np.mean([G.nodes[neighbor]['polarity'] for neighbor in neighbors])
                else:
                    avg = 0  # Default value if no neighbors match

                if node_value == pos_value:
                    pos_corr.append(avg)
                    pos_polarities.append(G.nodes[node]['polarity'])
                else:
                    neg_corr.append(avg)
                    neg_polarities.append(G.nodes[node]['polarity'])

        # Create marginal axes
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        ax_histx = divider.append_axes("top", size=0.35, pad=0, sharex=ax)
        ax_histy = divider.append_axes("right", size=0.35, pad=0, sharey=ax)

        # Scatter plot for positive and negative values
        ax.scatter(pos_polarities, pos_corr, c='darkred', s=20, alpha=0.95, label='ECC-P')
        ax.scatter(neg_polarities, neg_corr, c='darkblue', s=20, alpha=0.95, label='ECC-N')

        # Set bold line for y=x
        ax.plot([-1, 1], [-1, 1], 'k--', label='y=x line', linewidth=3)

        # Calculate average vertical distance to y=x line for all nodes
        distances = np.abs(np.array(pos_corr + neg_corr) - np.array(pos_polarities + neg_polarities))
        avg_distance = np.mean(distances) if len(distances) > 0 else 0

        # Annotations
        ax.annotate(f'Avg. distance: {avg_distance:.3f}',
            xy=(0.05, 0.9),
            xycoords='axes fraction',
            fontsize=14,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # Customize main scatter plot
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])

        # Reduce grid density by setting fewer ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(True, linestyle='--', alpha=0.5)

        ax.set_xlabel(method_names[method_name])
        # ax.set_ylabel('Average Neighbor Polarity')

        # Add legend only to the first subplot to avoid repetition
        if ax == axs[3]:
            legend = ax.legend(loc='lower right',
                frameon=True,
                framealpha=0.9)
            legend.get_frame().set_facecolor('white')
        else:
            ax.legend().set_visible(False)

        # Plot density for pos_polarities and neg_polarities on the top histogram
        if pos_polarities:
            sns.kdeplot(x=pos_polarities, ax=ax_histx, color='darkred', fill=True, alpha=0.5)
        if neg_polarities:
            sns.kdeplot(x=neg_polarities, ax=ax_histx, color='darkblue', fill=True, alpha=0.5)
        ax_histx.axis('off')  # Hide axis lines

        # Plot density for pos_corr and neg_corr on the right histogram
        if pos_corr:
            sns.kdeplot(y=pos_corr, ax=ax_histy, color='darkred', fill=True, alpha=0.5)
        if neg_corr:
            sns.kdeplot(y=neg_corr, ax=ax_histy, color='darkblue', fill=True, alpha=0.5)
        ax_histy.axis('off')  # Hide axis lines

    plt.tight_layout()
    # adjust the space between subplots
    plt.subplots_adjust(wspace=0.1)

    # Save as PDF if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


#compute the ratio of the number of edges between apposite communities
def border_stat(graph_path, value_pos=1, value_nag=-1, save_path=None):
    G = nx.read_gml(graph_path)
    method_names = {
        'maxflow_cpp_udsp': 'MaxFlow-U',
        'maxflow_cpp_wdsp': 'MaxFlow-W',
        'node2vec_gin': 'GIN',
        'pads_cpp': 'PADS'}
    df = pd.DataFrame(columns=['Method', 'OIR', 'BR-P', 'BR-N', 'Avg. Distance', 'RWC'])

    for attribute in method_names.keys():
        out = 0
        in_pos = 0
        in_nag = 0
        border_pos = set()
        border_nag = set()

        for edge in G.edges:
            if G.nodes[edge[0]][attribute] == value_pos and G.nodes[edge[1]][attribute] == value_nag:
                out += 1
                border_pos.add(edge[0])
                border_nag.add(edge[1])
            elif G.nodes[edge[0]][attribute] == value_nag and G.nodes[edge[1]][attribute] == value_pos:
                out += 1
                border_pos.add(edge[1])
                border_nag.add(edge[0])
            elif G.nodes[edge[0]][attribute] == value_pos and G.nodes[edge[1]][attribute] == value_pos:
                in_pos += 1
            elif G.nodes[edge[0]][attribute] == value_nag and G.nodes[edge[1]][attribute] == value_nag:
                in_nag += 1

        # More efficient distance calculation
        pos_nodes = [n for n, d in G.nodes(data=True) if d[attribute] == value_pos]
        neg_nodes = [n for n, d in G.nodes(data=True) if d[attribute] == value_nag]
        distances = [nx.shortest_path_length(G, s, t) for s in pos_nodes for t in neg_nodes]
        avg_distance = sum(distances)/len(distances) if distances else 0
        rwc_score = 1 if not (border_pos and border_nag) else rwc(G.copy(), attribute)

        count_pos = len(pos_nodes)
        count_nag = len(neg_nodes)

        df.loc[len(df)] = {
            'Method': method_names[attribute],
            'OIR': out/(in_pos + in_nag) if (in_pos + in_nag) > 0 else 0,
            'BR-P': len(border_pos)/count_pos if count_pos > 0 else 0,
            'BR-N': len(border_nag)/count_nag if count_nag > 0 else 0,
            'Avg. Distance': avg_distance,
            'RWC': rwc_score
        }

    if save_path is not None:
        df.to_csv(save_path, index=False)


def radar_chart(file_path, save_path):
    # Define the list of files and their corresponding titles
    files = [
        ("Abortion", "Abortion"),
        ("Brexit", "Brexit"),
        ("Election", "Election"),
        ("Gun", "Gun"),
        ("Partisanship", "Partisanship"),
        ("Referendum_", "Referendum"),
    ]

    # Define the categories (metrics)
    categories = ['OIR', 'BR-P', 'BR-N', 'RWC', 'Avg. Distance']
    num_vars = len(categories)

    # Define the order of methods for consistency
    method_names = ['MaxFlow-U', 'MaxFlow-W', 'GIN', 'PADS']

    # Define a more appealing color palette using matplotlib's tab10 via plt.get_cmap
    color_palette = plt.get_cmap('tab10')  # Updated to fix deprecation warning
    colors = color_palette.colors[:len(method_names)]  # Assign distinct colors
    # colors = ['g', 'c', 'y', 'm']  # Blue, Red, Green, Magenta

    # Create a figure with 2 rows and 3 columns of subplots, each polar
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), subplot_kw=dict(polar=True))
    axes = axes.flatten()  # Flatten the 2D array for easy iteration

    # Iterate over each file and corresponding subplot
    for idx, (file, title) in enumerate(files):
        ax = axes[idx]

        try:
            # Read the CSV file
            df = pd.read_csv(os.path.join(file_path, file, 'border_stat.csv'))

            # Ensure that the DataFrame has the methods in the specified order
            df = df.set_index('Method').loc[method_names].reset_index()
        except KeyError as e:
            print(f"Error: {e}. Please ensure all methods are present in {file}.")
            # Display a placeholder text in the subplot
            ax.text(0.5, 0.5, 'Data Missing', horizontalalignment='center',
                verticalalignment='center', fontsize=12, transform=ax.transAxes)
            # Set the title at the bottom
            ax.set_title(title, size=16, y=-0.15, fontsize=14, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            continue  # Skip this file if there's an error
        except FileNotFoundError:
            print(f"Error: {file} not found.")
            ax.text(0.5, 0.5, 'File Not Found', horizontalalignment='center',
                verticalalignment='center', fontsize=12, transform=ax.transAxes)
            # Set the title at the bottom
            ax.set_title(title, size=16, y=-0.15, fontsize=14, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            continue  # Skip this file if it's not found

        # Create a copy of the DataFrame for scaling
        scaled_df = df.copy()

        # Scaling the first three columns where smaller is better using min_val / x or 1 - x
        for col in ['OIR', 'BR-P', 'BR-N']:
            min_val = df[col].min()
            if min_val != 0:
                # Avoid division by zero
                scaled_df[col] = df[col].apply(lambda x: min_val / x if x != 0 else 1)
            else:
                # When min_val is 0
                # 0/0 = 1, and 0/x = 1 - x for x != 0
                scaled_df[col] = df[col].apply(lambda x: 1 if x == 0 else 1 - x)

        # Scaling the last column where larger is better using value / max_val
        for col in ['Avg. Distance', 'RWC']:
            max_val = df[col].max()
            if max_val != 0:
                scaled_df[col] = df[col].apply(lambda x: x / max_val)
            else:
                scaled_df[col] = 1  # Assign 1 if max_val is 0 to avoid NaN

        # Replace any infinite values resulted from division by zero with 1
        scaled_df.replace([np.inf, -np.inf], 1, inplace=True)

        # Replace NaN values with 1 as per your specification
        scaled_df.fillna(1, inplace=True)

        # Prepare the angles for the radar chart
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        # Plot each method
        for i, row in scaled_df.iterrows():
            method = row['Method']
            values = row[categories].tolist()
            values += values[:1]  # Complete the loop
            ax.plot(angles, values, color=colors[i], label=method, linewidth=2, linestyle='solid')
            ax.fill(angles, values, color=colors[i], alpha=0.2)

        # Configure the axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)

        # Set the radial limit to slightly above the maximum scaled value for better visualization
        max_scaled = scaled_df[categories].max().max()
        ax.set_ylim(0, max_scaled)  # Add a 20% margin

        # Configure the radial labels
        ax.set_rlabel_position(30)
        ax.set_yticks(np.linspace(0, max_scaled, 5))
        ax.set_yticklabels([f"{tick:.2f}" for tick in np.linspace(0, max_scaled, 5)], fontsize=10)

        # Add a title to the subplot at the bottom
        ax.set_title(title, size=16, y=-0.16, fontsize=14)#, fontweight='bold')

        # Optional: Add grid lines for better readability
        ax.grid(True, linestyle='--', linewidth=2, c='k', alpha=0.5)

    # Handle any unused subplots (if any)
    for j in range(len(files), len(axes)):
        fig.delaxes(axes[j])

    legend_elements = [Patch(facecolor=colors[i], label=method) for i, method in enumerate(method_names)]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(method_names),
        fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.95))

    # Adjust layout to prevent overlap and ensure titles and legends fit well
    # plt.tight_layout(rect=[0, 0, 1, 0.90], pad=2)

    # Enhance overall aesthetics
    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['figure.facecolor'] = 'white'

    # save the plot
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'radar.pdf'), bbox_inches='tight')

    # Display the figure
    plt.show()


def create_heatmap(graph_path, diffusion_path, save_path=None, grid_size=10):
    methods = {'maxflow_cpp_udsp': 'MaxFlow-U', 'maxflow_cpp_wdsp': 'MaxFlow-W', 'node2vec_gin': 'GIN', 'pads_cpp': 'PADS'}
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    axs = axs.flatten()
    for idx, m in enumerate(methods.keys()):
        G = nx.read_gml(graph_path)
        diffusion = json.load(open(diffusion_path))
        pos_nodes = [node for node in G.nodes if G.nodes[node][m] == 1]
        neg_nodes = [node for node in G.nodes if G.nodes[node][m] == -1]
        data = []
        for pos in pos_nodes:
            for neg in neg_nodes:
                data.append((G.nodes[pos]['polarity'], G.nodes[neg]['polarity'], diffusion['pos'][pos][int(neg)]))
            for pos in pos_nodes:
                data.append((G.nodes[pos]['polarity'], G.nodes[pos]['polarity'], diffusion['pos'][pos][int(pos)]))

        for neg in neg_nodes:
            for pos in pos_nodes:
                data.append((G.nodes[neg]['polarity'], G.nodes[pos]['polarity'], diffusion['neg'][neg][int(pos)]))
            for neg in neg_nodes:
                data.append((G.nodes[neg]['polarity'], G.nodes[neg]['polarity'], diffusion['neg'][neg][int(neg)]))

        grid_sum = np.zeros((grid_size, grid_size))
        grid_count = np.zeros((grid_size, grid_size))
        data = np.array(data)

        # Map each point to grid cell and accumulate values
        for x, y, p in data:
            # Convert from [-1,1] to [0,grid_size-1]
            grid_x = int((x + 1) * grid_size / 2)
            grid_y = int((y + 1) * grid_size / 2)

            # Handle edge cases
            grid_x = min(grid_x, grid_size - 1)
            grid_y = min(grid_y, grid_size - 1)

            grid_sum[grid_y, grid_x] += p
            grid_count[grid_y, grid_x] += 1

        # Calculate average, handling division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap = np.power(grid_sum / grid_count, 0.5)
        heatmap = np.nan_to_num(heatmap, 0)    # Replace NaN with 0

        # Apply Gaussian smoothing
        # heatmap = gaussian_filter(heatmap, sigma=0.1)
        # Create the plot
        axs[idx].imshow(heatmap, extent=[-1, 1, -1, 1], origin='lower', cmap='YlOrRd')
        # axs[idx].set_xlabel('Source Polarity')
        # axs[idx].set_ylabel('Target Polarity')
        # axs[idx].set_title(methods[m])
        axs[idx].set_xlabel(methods[m], fontsize=16)#, fontweight='bold')
        # axs[idx].set_xticks(np.linspace(-1, 1, grid_size + 1), fontsize=8)
        # set xticks invisible
        axs[idx].set_xticks([-1, 0, 1], labels=['-1', '0', '1'], fontsize=14)
        axs[idx].set_yticks([0, 1], labels=['0', '1'], fontsize=14)

    # plt.tight_layout()
    if save_path:
        # format pdf, no margin
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, format='pdf')


def fs_curve(save_path=None):
    datasets = {
                'Abortion': 'Abortion',
                'Brexit': 'Brexit',
                'Election': 'Election',
                'Gun': 'Gun',
                'Partisanship': 'Partisanship',
                'Referendum_': 'Referendum'
                }
    fig, axs = plt.subplots(2, 3, figsize=(8, 6))
    axs = axs.flatten()

    for idx, d in enumerate(datasets.keys()):
        G = get_graph(d)
        pos, neg = pads_python(G, return_fs=True)
        pos_values = [p[0] for p in pos]
        neg_values = [n[0] for n in neg]

        # Plot curves
        axs[idx].plot(pos_values)
        axs[idx].plot(neg_values)

        # Find peaks
        pos_peak = max(pos_values)
        pos_peak_idx = pos_values.index(pos_peak)
        neg_peak = max(neg_values)
        neg_peak_idx = neg_values.index(neg_peak)

        # Add red stars at peaks
        axs[idx].plot(pos_peak_idx, pos_peak, 'r*', markersize=6)
        axs[idx].plot(neg_peak_idx, neg_peak, 'r*', markersize=6)

        # Add vertical and horizontal lines for positive peak
        axs[idx].vlines(x=pos_peak_idx, ymin=0, ymax=pos_peak,
            colors='r', linestyles='--', alpha=0.5)
        axs[idx].hlines(y=pos_peak, xmin=0, xmax=pos_peak_idx,
            colors='r', linestyles='--', alpha=0.5)

        # Add vertical and horizontal lines for negative peak
        axs[idx].vlines(x=neg_peak_idx, ymin=0, ymax=neg_peak,
            colors='r', linestyles='--', alpha=0.5)
        axs[idx].hlines(y=neg_peak, xmin=0, xmax=neg_peak_idx,
            colors='r', linestyles='--', alpha=0.5)

        axs[idx].set_xlabel(datasets[d], fontsize=10)
        axs[idx].grid(True, linestyle='--', alpha=0.3)
        # set y ticks invisible
        axs[idx].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        # set x ticks smaller
        axs[idx].tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.show()


def theta_influence(dataset, save_path=None, thetas=[0, 0.25, 0.5, 1, 2, 4]):
    # Set Seaborn style for better aesthetics
    sns.set(style="whitegrid")

    G = get_graph(dataset)
    y_labels = ['f(S)', 'Weight Sum', '# Nodes', 'Entropy', 'Density']
    times = {}
    pos_fss = {}
    neg_fss = {}

    # Run experiments for each theta
    for theta in thetas:
        # timer = {}
        t, (pos_fs, neg_fs) = run_exp(G, 'pads_python', theta=theta, return_fs=True)
        times[theta] = t
        pos_fss[theta] = pos_fs
        neg_fss[theta] = neg_fs

    # Print time consumption and results
    print(f"=== Time Consumption ===\n{times}")
    print(f"=== Results ===")
    # print(f"Positive Influence: {pos_fss}")
    # print(f"Negative Influence: {neg_fss}")

    # Prepare color palette
    palette = sns.color_palette("viridis", n_colors=len(thetas))

    # Create subplots with constrained layout
    fig, axs = plt.subplots(2, 5, figsize=(15, 5.5), constrained_layout=True, sharex=True)
    axs = axs.flatten()

    # Initialize lists for custom legend
    legend_elements = []

    # Define line styles for better distinction (optional)
    line_styles = ['-', '--', '-.', ':', '-', '--']

    for i in range(5):
        ax_pos = axs[i]
        ax_neg = axs[i + 5]

        for idx, theta in enumerate(thetas):
            color = palette[idx]
            line_style = line_styles[idx % len(line_styles)]  # Cycle through line styles

            if i == 4:
                pos_values = [v[1]/v[2] for v in pos_fss[theta]]
                neg_values = [v[1]/v[2] for v in neg_fss[theta]]
            else:
                pos_values = [v[i] for v in pos_fss[theta]]
                neg_values = [v[i] for v in neg_fss[theta]]

            ax_pos.plot(pos_values, label=f"θ={theta}", color=color, linestyle=line_style, linewidth=2)
            ax_neg.plot(neg_values, label=f"θ={theta}", color=color, linestyle=line_style, linewidth=2)

        # Add grid with subtle styling
        ax_pos.grid(True, linestyle='--', alpha=0.3)
        ax_neg.grid(True, linestyle='--', alpha=0.3)

        # Set y labels
        ax_pos.set_ylabel(y_labels[i], fontsize=12)#, fontweight='bold')
        ax_neg.set_ylabel(y_labels[i], fontsize=12)#, fontweight='bold')

        # Set x labels on the bottom row
        # ax_neg.set_xlabel('#Addition/Removal', fontsize=12, fontweight='bold')

        # Add titles for the first subplot in each row
        # if i == 0:
        #     ax_pos.set_title('Positive Influence', fontsize=14, fontweight='bold')
        #     ax_neg.set_title('Negative Influence', fontsize=14, fontweight='bold')

        # Adjust tick parameters for clarity
        ax_pos.tick_params(axis='both', which='major', labelsize=10)
        ax_neg.tick_params(axis='both', which='major', labelsize=10)

        # Dynamic axis scaling to fit data tightly
        ax_pos.relim()
        ax_pos.autoscale_view()
        ax_neg.relim()
        ax_neg.autoscale_view()

        # Reduce margins to make data occupy more space
        # ax_pos.margins(x=0.02, y=0.05)
        # ax_neg.margins(x=0.02, y=0.05)

    # Create a single legend for all subplots
    for idx, theta in enumerate(thetas):
        legend_elements.append(Line2D([0], [0], color=palette[idx], linestyle=line_styles[idx % len(line_styles)], lw=2, label=f'θ={theta}'))

    # Place legend centrally below all subplots
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(thetas), bbox_to_anchor=(0.5, 1), fontsize=12)

    # Add a main title
    # fig.suptitle('Theta Influence on Positive and Negative Metrics', fontsize=18, fontweight='bold', y=0.98)

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf')

    # Show the plot
    plt.show()

def plot_opinion_dynamics(datasets=['Referendum_', 'Gun'], save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs = axs.flatten()
    for i, d in enumerate(datasets):
        opinion_dynamics_reweight(d, ax=axs[i])
        axs[i].set_xlabel('Time Steps')
        axs[i].set_ylabel('Polarity Variance')
        axs[i].set_title(d.replace("_", ""))
    axs[-1].legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


def polarity_distance(dataset, ax, color_pos='#ff7f0e', color_neg='#1f77b4'):
    G = nx.read_gml(f'Output/{dataset}/graph.gml')

    for i in [1, -1]:
        label = 'Positive' if i==1 else 'Negative'
        color = color_pos if i==1 else color_neg

        # Find border nodes more efficiently
        borders = [node for node in G.nodes
            if G.nodes[node]['pads_cpp'] == i and
               any(G.nodes[n]['pads_cpp'] == 0 for n in G.neighbors(node))]

        if not borders:
            continue

        # Calculate distances once and store results
        distance_data = defaultdict(list)

        # Calculate distances from all border nodes at once
        for node in G.nodes():
            min_dist = float('inf')
            for b in borders:
                try:
                    dist = nx.shortest_path_length(G, b, node)
                    min_dist = min(min_dist, dist)
                except nx.NetworkXNoPath:
                    continue

            if min_dist != float('inf'):
                if G.nodes[node]['pads_cpp'] == i:
                    min_dist *= -1
                if min_dist != float('inf'):
                    distance_data[min_dist].append(G.nodes[node]['polarity'])

        # Process the collected data
        distances = []
        polarities = []
        sizes = []

        for dist in sorted(distance_data.keys()):
            pol_values = distance_data[dist]
            if pol_values:
                distances.append(dist)
                polarities.append(sum(pol_values) / len(pol_values))
                sizes.append(len(pol_values))

        if not distances:  # Skip if no valid data
            continue

        # Plot scatter points
        sizes_normalized = [50 * s / max(sizes) for s in sizes]
        ax.scatter(distances, polarities, s=sizes_normalized, alpha=0.5, color=color)

        # Simple line plot instead of curve fitting
        ax.plot(distances, polarities, color=color, label=f'{label} ECC', alpha=0.7)

    ax.set_xlabel('Distance')
    ax.set_ylabel('Average Polarity')
    ax.grid(True, alpha=0.3)
    ax.legend()

# Create subplot grid
def plot_polarity_distance(datasets):
    n_datasets = len(datasets)
    n_cols = 2
    n_rows = (n_datasets + 1) // 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axs = axs.flatten() if n_datasets > 1 else [axs]

    # Plot each dataset
    for idx, d in enumerate(datasets):
        # axs[idx].set_title(f'Dataset {d}')
        polarity_distance(d, axs[idx])

    # Remove empty subplots if odd number of datasets
    if n_datasets > 1 and n_datasets % 2 == 1:
        fig.delaxes(axs[-1])

    plt.tight_layout()
    plt.show()