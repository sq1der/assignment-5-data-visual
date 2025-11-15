# Assignment 5: Open3D 3D Model Processing Pipeline

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Open3D](https://img.shields.io/badge/Open3D-0.18+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive 3D model processing pipeline using Open3D library that demonstrates mesh manipulation, point cloud processing, surface reconstruction, and visualization techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Steps](#pipeline-steps)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Output Examples](#output-examples)
- [Troubleshooting](#troubleshooting)
- [Author](#author)

## 🎯 Overview

This project implements a complete 3D geometry processing pipeline that covers fundamental operations in computational geometry and 3D graphics:

- **Mesh Loading**: Import various 3D file formats
- **Point Cloud Generation**: Sample points from mesh surfaces
- **Surface Reconstruction**: Rebuild surfaces using Poisson reconstruction
- **Voxelization**: Convert geometry to volumetric representation
- **Geometric Operations**: Plane addition and surface clipping
- **Visualization**: Color gradients and extreme point detection

## ✨ Features

- ✅ **Multi-format Support**: `.ply`, `.obj`, `.stl`, `.off` files
- ✅ **Step-by-step Visualization**: Each operation displays results immediately
- ✅ **Detailed Statistics**: Comprehensive information about geometry properties
- ✅ **Clean Code Architecture**: Modular functions with clear documentation
- ✅ **Professional Output**: Beautiful formatted console output
- ✅ **Error Handling**: Graceful handling of common issues

## 📦 Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for large models)
- OpenGL-compatible graphics card

### Python Libraries
```
open3d >= 0.18.0
numpy >= 1.20.0
```

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/assignment-5.git
cd assignment-5
```

### 2. Create Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install open3d numpy
```

### 4. Download 3D Model
Place your 3D model file in the `models/` directory. You can download free models from:
- [Free3D](https://free3d.com/3d-models)
- [Sketchfab](https://sketchfab.com/features/download)
- [TurboSquid Free](https://www.turbosquid.com/Search/3D-Models/free)

**Important**: Each student must use a unique 3D model!

## 💻 Usage

### Basic Usage
```bash
python3 main_auto.py --input /path/to/your/model
```

## 🔄 Pipeline Steps

### Step 1: Load and Visualize 📂
- Loads 3D model from file
- Computes vertex normals if missing
- Displays basic mesh statistics
- **Output**: Original mesh visualization

### Step 2: Point Cloud Conversion ☁️
- Converts mesh to point cloud
- Samples points uniformly from surface
- Preserves color information if available
- **Output**: Point cloud visualization

### Step 3: Surface Reconstruction 🏗️
- Estimates point cloud normals
- Performs Poisson surface reconstruction
- Removes low-density artifacts
- **Output**: Reconstructed mesh

### Step 4: Voxelization 🧊
- Converts point cloud to voxel grid
- Creates volumetric representation
- Configurable voxel size
- **Output**: Voxel grid visualization

### Step 5: Plane Addition ➕
- Creates geometric plane
- Positions plane relative to mesh
- Prepares for clipping operation
- **Output**: Mesh with plane

### Step 6: Surface Clipping ✂️
- Defines clipping plane
- Removes vertices on one side
- Preserves colors and normals
- **Output**: Clipped geometry

### Step 7: Color Gradient & Extremes 🌈
- Applies blue-to-red gradient along Z-axis
- Finds minimum and maximum points
- Marks extremes with colored spheres
- **Output**: Gradient visualization with markers

## 📁 Project Structure

```
assignment-5/
│
├── main.py                 # Main script
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
└──  models/                   # 3D model files
    └── your_model.ply
```

## ⚙️ Configuration

### Adjusting for Different Models

**Small models** (< 10K vertices):
```python
POINT_CLOUD_SAMPLES = 5000
VOXEL_SIZE = 0.1
POISSON_DEPTH = 8
```

**Large models** (> 1M vertices):
```python
POINT_CLOUD_SAMPLES = 50000
VOXEL_SIZE = 0.02
POISSON_DEPTH = 10
```

## 📊 Output Examples

### Console Output
```
🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 
OPEN3D 3D MODEL PROCESSING PIPELINE - ASSIGNMENT 5
🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 

======================================================================
STEP 1: LOADING AND VISUALIZATION
======================================================================
├─ Number of vertices: 1,170,210
├─ Number of triangles: 6,524
├─ Has vertex colors: False
└─ Has vertex normals: True

🖼️  Displaying: Step 1: Original Mesh
...
```

### Visual Output
Each step opens a new visualization window with:
- 3D interactive view
- Mouse controls (rotate, zoom, pan)
- Default lighting and shading

## 🐛 Troubleshooting

### Common Issues

#### Issue: "Read PLY failed: A polygon could not be decomposed"
**Solution**: This is a warning, not an error. Open3D successfully loads the model using an alternative method.

#### Issue: Empty visualization window
**Solution**: 
```bash
# macOS
export DISPLAY=:0

# Linux - install mesa
sudo apt-get install libgl1-mesa-glx
```

#### Issue: Out of memory error
**Solution**: Reduce `POINT_CLOUD_SAMPLES` or `POISSON_DEPTH`
```python
POINT_CLOUD_SAMPLES = 5000
POISSON_DEPTH = 7
```

#### Issue: Model appears too small/large
**Solution**: Adjust visualization settings or scale the model
```python
mesh.scale(10, center=mesh.get_center())  # Scale up
```

### Getting Help

If you encounter issues:
1. Check that your Python version is 3.8+
2. Verify Open3D installation: `python -c "import open3d; print(open3d.__version__)"`
3. Try a different 3D model file
4. Check file permissions for model directory

## 📚 Resources

### Documentation
- [Open3D Documentation](http://www.open3d.org/docs/release/)
- [Open3D Python Tutorial](http://www.open3d.org/docs/release/tutorial/index.html)
- [NumPy Documentation](https://numpy.org/doc/)

### 3D Model Sources
- [Free3D](https://free3d.com/3d-models)
- [Sketchfab](https://sketchfab.com)
- [TurboSquid Free](https://www.turbosquid.com/Search/3D-Models/free)
- [Thingiverse](https://www.thingiverse.com)

### Learning Resources
- [3D Computer Graphics - Wikipedia](https://en.wikipedia.org/wiki/3D_computer_graphics)
- [Point Cloud Processing](https://www.cloudcompare.org/doc/wiki/index.php?title=Main_Page)
- [Surface Reconstruction](https://en.wikipedia.org/wiki/Surface_reconstruction)

## 👤 Author

**[Your Name]**
- Student ID: [Your ID]
- Course: Computer Graphics / 3D Modeling
- Assignment: #5
- Date: November 2025

## 📝 Assignment Requirements

### Grading Criteria
- ✅ All 7 steps completed: **100 points**
- ❌ Missing even one step: **0 points**

### Submission Checklist
- [ ] GitHub repository updated
- [ ] Unique 3D model used (not same as other students)
- [ ] All scripts run successfully
- [ ] Console output shows all required information
- [ ] Screenshots NOT uploaded to GitHub
- [ ] README.md included
- [ ] Code is clean and well-documented

### Defense Requirements
- ✅ Live script execution (no screenshots)
- ✅ Explain each step's purpose
- ✅ Show understanding of:
  - Vertices vs triangles
  - Point clouds vs meshes
  - Voxelization concept
  - Clipping operations
  - Coordinate systems

## 📄 License

This project is created for educational purposes as part of Assignment 5.

---

**Note**: This assignment must be completed individually. Code sharing or plagiarism will result in a grade of 0 points.

## 🎓 Tips for Success

1. **Test Early**: Run your script multiple times before submission
2. **Understand, Don't Memorize**: Know what each line does
3. **Document Changes**: Comment any modifications you make
4. **Ask Questions**: Clarify doubts before the deadline
5. **Backup Your Work**: Commit to GitHub regularly

---

*Last Updated: November 2025*
*Version: 1.0*

**Good luck with your assignment! 🚀**
