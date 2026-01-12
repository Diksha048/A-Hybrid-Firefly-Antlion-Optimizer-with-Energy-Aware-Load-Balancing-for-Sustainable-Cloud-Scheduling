# Hybrid Firefly-Antlion Optimizer (FAO)

Energy-Efficient Cloud Task Scheduling Web Application

## 📁 Project Structure

```
hybrid-fao-webapp/
├── app.py                      # Main Flask application
├── model.py                    # Optimization algorithms
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── static/
│   ├── css/
│   │   └── style.css          # Main stylesheet
│   ├── js/
│   │   └── main.js            # JavaScript utilities
│   └── images/
│       └── logo.svg           # Logo (optional)
├── templates/
│   ├── base.html              # Base template
│   ├── home.html              # Home page
│   ├── dashboard.html         # Interactive dashboard
│   ├── results.html           # Results comparison
│   ├── model.html             # Model explanation
│   ├── impact.html            # Impact metrics
│   └── about.html             # About page
└── data/
    ├── tasks_dataset.csv      # Tasks dataset
    └── vms_dataset.csv        # VMs dataset
```

## 🚀 Setup Instructions

### 1. Create Project Directory

```bash
mkdir hybrid-fao-webapp
cd hybrid-fao-webapp
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Data

Create a `data/` folder and place your CSV files:
- `tasks_dataset.csv`
- `vms_dataset.csv`

Make sure your CSV files have the required columns as shown in your code.

### 5. Create Folder Structure

```bash
# Create necessary directories
mkdir static static/css static/js static/images
mkdir templates
mkdir data
```

### 6. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## 📊 Features

- **Home Page**: Landing page with project overview
- **Dashboard**: Interactive controls to run optimizations
- **Results**: Compare algorithm performance
- **Model**: Explanation of the hybrid approach
- **Impact**: Environmental and cost impact metrics
- **About**: Team information

## 🎯 Usage

1. Navigate to the **Dashboard** page
2. Set the number of tasks and VMs
3. Select an algorithm (Firefly, AntLion, or Hybrid)
4. Adjust weight parameters using sliders
5. Click "Run Optimization" to see results
6. View detailed comparisons on the **Results** page

## 🔧 API Endpoints

- `POST /api/run-optimization` - Run optimization with custom parameters
- `GET /api/get-comparison` - Get comparison data for all algorithms

## 🔌 Backend Integration Guide

### 1. Flask Application Setup
- Configure the main web application framework to handle HTTP requests and responses
- Set up routing for different pages and API endpoints
- Implement data loading and preprocessing for task and virtual machine datasets
- Establish global variables for sharing data between optimization runs

### 2. Optimization Model Integration
- Import the three optimization algorithm classes (Firefly, Ant Lion, and Hybrid)
- Implement fitness calculation functions for evaluating solution quality
- Create metric calculation utilities for performance analysis
- Set up data preprocessing to handle task priorities and arrival times

### 3. API Endpoints Implementation
- Create endpoints for running individual algorithm optimizations
- Implement comparison functionality to run all algorithms simultaneously
- Add validation for input parameters (task count, VM count, algorithm selection)
- Handle error responses and success confirmations

## 🎨 Customization

### Colors
Edit `static/css/style.css` to modify the color scheme:
```css
:root {
    --bg-primary: #0a1628;
    --accent-primary: #2dd4bf;
    /* ... more variables */
}
```

### Algorithms
Modify `model.py` to adjust algorithm parameters or add new optimization methods.

## 📝 Notes

- The application uses the exact algorithms from your provided code
- Results are calculated in real-time
- All visualizations match the design from your images
- The UI is fully responsive and works on all devices

## 🐛 Troubleshooting

**Issue**: "Module not found" error
- **Solution**: Make sure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: "File not found" error for CSV
- **Solution**: Ensure CSV files are in the `data/` directory with correct names

**Issue**: Flask app won't start
- **Solution**: Check if port 5000 is available or change the port in `app.py`

## 📦 Deployment

For production deployment, consider using:
- **Gunicorn**: Production WSGI server
- **Nginx**: Reverse proxy
- **Docker**: Containerization

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## 👥 Contributors

- João Silva - PhD Candidate, Computer Science
- Maria Oliveira - Master's Student, Computer Science
- Ravi Patel - Master's Student, Electrical Engineering

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

Based on research in hybrid optimization algorithms for cloud task scheduling.