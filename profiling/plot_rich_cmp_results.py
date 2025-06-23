#!/usr/bin/env python3
"""
Plot performance comparison results from profile_rich_cmp.output
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_results_file(file_path):
    """Parse the rich comparison results file and extract data."""
    data = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split by data types
    sections = re.split(r'=== Testing (\w+) ===', content)[1:]
    
    for i in range(0, len(sections), 2):
        data_type = sections[i]
        section_content = sections[i + 1]
        
        data[data_type] = {}
        
        # Extract each N value section
        pattern = (r'## N = (\w+) type: \w+\n\n.*?\| np\s+\| '
                   r'([\d\.E\-\+]+)\s+\|.*?\| sa\s+\| ([\d\.E\-\+]+)\s+\|')
        n_sections = re.findall(pattern, section_content, re.DOTALL)
        
        for n_size, np_time, sa_time in n_sections:
            # Convert size notation to number
            if n_size.endswith('K'):
                n_val = int(n_size[:-1]) * 1024
            elif n_size.endswith('M'):
                n_val = int(n_size[:-1]) * 1024 * 1024
            else:
                n_val = int(n_size)
            
            data[data_type][n_val] = {
                'np': float(np_time),
                'sa': float(sa_time)
            }
    
    return data


def plot_performance_comparison(data, save_path=None):
    """Create performance comparison plots."""
    # Create figure with subplots
    n_types = len(data)
    cols = 3
    rows = (n_types + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (data_type, type_data) in enumerate(data.items()):
        ax = axes[idx]
        
        # Prepare data for plotting
        sizes = sorted(type_data.keys())
        np_times = [type_data[size]['np'] for size in sizes]
        sa_times = [type_data[size]['sa'] for size in sizes]
        
        # Create size labels
        size_labels = []
        for size in sizes:
            if size >= 1024 * 1024:
                size_labels.append(f"{size // (1024 * 1024)}M")
            elif size >= 1024:
                size_labels.append(f"{size // 1024}K")
            else:
                size_labels.append(str(size))
        
        x = np.arange(len(sizes))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, np_times, width, label='NumPy',
                       alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, sa_times, width, label='SimpleArray',
                       alpha=0.8, color='lightcoral')
        
        # Customize plot
        ax.set_xlabel('Array Size')
        ax.set_ylabel('Time per call (ms)')
        ax.set_title(f'Performance Comparison - {data_type}')
        ax.set_xticks(x)
        ax.set_xticklabels(size_labels, rotation=45)
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars (simplified for readability)
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1e}', ha='center', va='bottom',
                    fontsize=7, rotation=0)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1e}', ha='center', va='bottom',
                    fontsize=7, rotation=0)
    
    # Hide unused subplots
    for idx in range(n_types, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_speedup_comparison(data, save_path=None):
    """Create speedup comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Group data types by category
    int_types = [t for t in data.keys() if 'int' in t and 'uint' not in t]
    uint_types = [t for t in data.keys() if 'uint' in t]
    float_types = [t for t in data.keys() if 'float' in t]
    bool_types = [t for t in data.keys() if t == 'bool']
    
    type_groups = [
        ('Integer Types', int_types),
        ('Unsigned Integer Types', uint_types),
        ('Float Types', float_types),
        ('Boolean Type', bool_types)
    ]
    
    for idx, (group_name, types) in enumerate(type_groups):
        if not types:
            continue
            
        ax = axes[idx]
        
        for data_type in types:
            type_data = data[data_type]
            sizes = sorted(type_data.keys())
            
            # Calculate speedup ratio (sa_time / np_time)
            speedup_ratios = []
            for size in sizes:
                np_time = type_data[size]['np']
                sa_time = type_data[size]['sa']
                speedup_ratios.append(sa_time / np_time)
            
            # Create size labels
            size_labels = []
            for size in sizes:
                if size >= 1024 * 1024:
                    size_labels.append(f"{size // (1024 * 1024)}M")
                elif size >= 1024:
                    size_labels.append(f"{size // 1024}K")
                else:
                    size_labels.append(str(size))
            
            ax.plot(size_labels, speedup_ratios, marker='o', label=data_type,
                    linewidth=2, markersize=6)
        
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5,
                   label='Equal Performance')
        ax.set_xlabel('Array Size')
        ax.set_ylabel('Time Ratio (SA/NP)')
        ax.set_title(f'{group_name}\n(< 1.0 means SA is faster)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Speedup plot saved to: {save_path}")
    
    plt.show()


def main():
    """Main function to parse data and create plots."""
    # Define file paths
    base_path = Path(__file__).parent / "results"
    results_file = base_path / "profile_rich_cmp.output"
    plot_output = base_path / "rich_cmp_performance.png"
    speedup_output = base_path / "rich_cmp_speedup.png"
    
    # Check if results file exists
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    
    # Parse the results
    print("Parsing results file...")
    data = parse_results_file(results_file)
    
    print(f"Found data for {len(data)} data types:")
    for data_type in data.keys():
        print(f"  - {data_type}: {len(data[data_type])} size configurations")
    
    # Create plots
    print("\nCreating performance comparison plots...")
    plot_performance_comparison(data, plot_output)
    
    print("\nCreating speedup comparison plots...")
    plot_speedup_comparison(data, speedup_output)
    
    print("\nPlotting completed!")


if __name__ == "__main__":
    main()
