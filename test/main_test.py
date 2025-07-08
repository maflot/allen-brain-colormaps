"""
Test code for Allen Brain Colormaps package
Run this to verify the package is working correctly
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Test imports
print("Testing imports...")
try:
    from allen_brain_colormaps import (
        get_brain_colors, 
        get_brain_cmap, 
        plot_brain_palette,
        AllenBrainColormaps,
        class_to_color,
        subclass_to_color,
        supertype_to_color
    )
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

def test_basic_functionality():
    """Test basic color and colormap functionality."""
    print("\n=== Testing Basic Functionality ===")
    
    # Test getting all colors
    print("Testing get_brain_colors()...")
    
    class_colors = get_brain_colors('class')
    subclass_colors = get_brain_colors('subclass') 
    supertype_colors = get_brain_colors('supertype')
    
    print(f"✅ Class colors: {len(class_colors)} types")
    print(f"✅ Subclass colors: {len(subclass_colors)} types")
    print(f"✅ Supertype colors: {len(supertype_colors)} types")
    
    # Test getting specific colors
    specific_colors = get_brain_colors('subclass', ['Astrocyte', 'Pvalb', 'Vip'])
    print(f"✅ Specific colors: {specific_colors}")
    
    # Test colormaps
    print("Testing get_brain_cmap()...")
    for level in ['class', 'subclass', 'supertype']:
        cmap = get_brain_cmap(level)
        print(f"✅ {level} colormap: {cmap.name}, {cmap.N} colors")

def test_matplotlib_integration():
    """Test matplotlib plotting functionality."""
    print("\n=== Testing Matplotlib Integration ===")
    
    # Basic bar plot
    cell_types = ['Astrocyte', 'Pvalb', 'Vip', 'Sst', 'L6 IT']
    values = [85, 72, 68, 55, 42]
    colors = get_brain_colors('subclass', cell_types)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cell_types, values, color=[colors[ct] for ct in cell_types])
    ax.set_title('Test: Cell Type Expression Levels')
    ax.set_ylabel('Expression')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    print("✅ Bar plot created successfully")
    plt.show()
    
    # Colormap with scatter plot
    cmap = get_brain_cmap('subclass')
    
    # Generate test data
    np.random.seed(42)
    n_points = 100
    x = np.random.randn(n_points)
    y = np.random.randn(n_points)
    c = np.random.randint(0, len(subclass_to_color), n_points)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(x, y, c=c, cmap=cmap, s=50, alpha=0.7)
    ax.set_title('Test: Scatter Plot with Brain Colormap')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    plt.colorbar(scatter, ax=ax, label='Cell Type Index')
    
    print("✅ Scatter plot with colormap created successfully")
    plt.show()

def test_seaborn_integration():
    """Test seaborn plotting functionality."""
    print("\n=== Testing Seaborn Integration ===")
    
    # Create sample dataset
    np.random.seed(42)
    cell_types = ['Astrocyte', 'Pvalb', 'Vip', 'Sst']
    n_samples = 50
    
    data = []
    for ct in cell_types:
        values = np.random.normal(loc=np.random.uniform(30, 80), scale=10, size=n_samples)
        for val in values:
            data.append({'cell_type': ct, 'expression': val})
    
    df = pd.DataFrame(data)
    
    # Get colors
    colors = get_brain_colors('subclass', cell_types)
    
    # Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='cell_type', y='expression', palette=colors)
    plt.title('Test: Seaborn Box Plot with Brain Colors')
    plt.ylabel('Expression Level')
    plt.xlabel('Cell Type')
    
    print("✅ Seaborn box plot created successfully")
    plt.show()
    
    # Violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='cell_type', y='expression', palette=colors)
    plt.title('Test: Seaborn Violin Plot with Brain Colors')
    plt.ylabel('Expression Level')
    plt.xlabel('Cell Type')
    
    print("✅ Seaborn violin plot created successfully")
    plt.show()

def test_palette_visualization():
    """Test palette visualization functions."""
    print("\n=== Testing Palette Visualization ===")
    
    # Test each hierarchy level
    for level in ['class', 'subclass', 'supertype']:
        print(f"Creating {level} palette...")
        
        # Adjust figure size based on number of items
        figsize = (12, 4) if level == 'class' else (12, 8) if level == 'subclass' else (12, 16)
        
        fig, ax = plot_brain_palette(level, figsize=figsize)
        plt.show()
        print(f"✅ {level} palette visualization successful")

def test_class_functionality():
    """Test AllenBrainColormaps class directly."""
    print("\n=== Testing AllenBrainColormaps Class ===")
    
    brain = AllenBrainColormaps()
    
    # Test class methods
    class_colors = brain.get_class_colors()
    subclass_colors = brain.get_subclass_colors(['Astrocyte', 'Pvalb'])
    supertype_colors = brain.get_supertype_colors(['Astro_1', 'Pvalb_1'])
    
    print(f"✅ Class colors: {len(class_colors)} types")
    print(f"✅ Specific subclass colors: {subclass_colors}")
    print(f"✅ Specific supertype colors: {supertype_colors}")
    
    # Test colormap creation
    cmap = brain.get_cmap('subclass')
    print(f"✅ Colormap creation: {cmap.name}")

def test_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Testing Error Handling ===")
    
    # Test invalid level
    try:
        get_brain_colors('invalid_level')
        print("❌ Should have raised ValueError for invalid level")
    except ValueError:
        print("✅ Correctly handled invalid level")
    
    # Test missing cell type
    colors = get_brain_colors('subclass', ['Astrocyte', 'NonexistentType'])
    expected_color = '#000000'  # Default color for missing types
    if colors['NonexistentType'] == expected_color:
        print("✅ Correctly handled missing cell type")
    else:
        print(f"❌ Unexpected color for missing type: {colors['NonexistentType']}")
    
    # Test empty cell type list
    colors = get_brain_colors('subclass', [])
    if len(colors) == 0:
        print("✅ Correctly handled empty cell type list")
    else:
        print("❌ Should return empty dict for empty list")

def test_color_consistency():
    """Test that colors are consistent across different access methods."""
    print("\n=== Testing Color Consistency ===")
    
    # Compare direct dictionary access vs function access
    astrocyte_color1 = subclass_to_color['Astrocyte']
    astrocyte_color2 = get_brain_colors('subclass', ['Astrocyte'])['Astrocyte']
    
    brain = AllenBrainColormaps()
    astrocyte_color3 = brain.get_subclass_colors(['Astrocyte'])['Astrocyte']
    
    if astrocyte_color1 == astrocyte_color2 == astrocyte_color3:
        print("✅ Colors consistent across access methods")
    else:
        print(f"❌ Color inconsistency: {astrocyte_color1} vs {astrocyte_color2} vs {astrocyte_color3}")

def test_real_world_example():
    """Test a real-world-like analysis example."""
    print("\n=== Testing Real-World Example ===")
    
    # Simulate single-cell data analysis
    np.random.seed(42)
    
    # Create mock single-cell data
    n_cells = 1000
    cell_types = ['Astrocyte', 'Pvalb', 'Vip', 'Sst', 'L6 IT', 'Microglia-PVM']
    
    # Generate cell type assignments
    cell_type_assignments = np.random.choice(cell_types, n_cells, 
                                           p=[0.3, 0.15, 0.15, 0.15, 0.15, 0.1])
    
    # Generate mock UMAP coordinates
    umap_x = np.random.randn(n_cells) * 2
    umap_y = np.random.randn(n_cells) * 2
    
    # Add some structure - cluster cell types
    for i, ct in enumerate(cell_types):
        mask = cell_type_assignments == ct
        umap_x[mask] += (i % 3) * 4
        umap_y[mask] += (i // 3) * 4
    
    # Create DataFrame
    df = pd.DataFrame({
        'UMAP_1': umap_x,
        'UMAP_2': umap_y,
        'cell_type': cell_type_assignments
    })
    
    # Get colors
    colors = get_brain_colors('subclass', cell_types)
    
    # Create publication-quality plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # UMAP plot
    for ct in cell_types:
        mask = df['cell_type'] == ct
        ax1.scatter(df.loc[mask, 'UMAP_1'], df.loc[mask, 'UMAP_2'], 
                   c=colors[ct], label=ct, alpha=0.6, s=20)
    
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.set_title('Mock Single-Cell UMAP')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Cell type counts
    counts = df['cell_type'].value_counts()
    ax2.bar(range(len(counts)), counts.values, 
           color=[colors[ct] for ct in counts.index])
    ax2.set_xticks(range(len(counts)))
    ax2.set_xticklabels(counts.index, rotation=45, ha='right')
    ax2.set_ylabel('Number of Cells')
    ax2.set_title('Cell Type Distribution')
    
    plt.tight_layout()
    print("✅ Real-world example plot created successfully")
    plt.show()

def run_all_tests():
    """Run all test functions."""
    print("🧪 Starting Allen Brain Colormaps Test Suite")
    print("=" * 50)
    
    test_functions = [
        test_basic_functionality,
        test_matplotlib_integration, 
        test_seaborn_integration,
        test_palette_visualization,
        test_class_functionality,
        test_error_handling,
        test_color_consistency,
        test_real_world_example
    ]
    
    failed_tests = []
    
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed_tests.append(test_func.__name__)
    
    print("\n" + "=" * 50)
    if failed_tests:
        print(f"❌ {len(failed_tests)} tests failed: {', '.join(failed_tests)}")
    else:
        print("✅ All tests passed successfully!")
    print("🎉 Test suite completed!")

if __name__ == "__main__":
    # You can run individual tests or all tests
    
    # Run all tests
    run_all_tests()
    
    # Or run individual tests:
    # test_basic_functionality()
    # test_matplotlib_integration()
    # test_seaborn_integration()
    # test_palette_visualization()
    # test_real_world_example()