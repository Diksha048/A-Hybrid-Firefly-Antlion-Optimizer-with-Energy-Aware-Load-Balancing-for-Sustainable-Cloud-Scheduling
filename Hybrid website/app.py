from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# Import the model implementations
from model import (
    FireflyAlgorithmFast,
    AntLionOptimizerFast,
    OptimizedHybridFast,
    calculate_fitness_fast,
    calculate_metrics,
    run_full_optimization
)
import model

app = Flask(__name__)

# Load datasets
DATA_DIR = 'data'
tasks_df = pd.read_csv(os.path.join(DATA_DIR, 'tasks_dataset.csv'))
vms_df = pd.read_csv(os.path.join(DATA_DIR, 'vms_dataset.csv'))

# Preprocess data
priority_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
tasks_df['Priority_Encoded'] = tasks_df['Priority'].map(priority_mapping)

# Global variables to store results
latest_results = {
    'firefly': None,
    'antlion': None,
    'hybrid': None
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/results')
def results():
    return render_template('results.html', results=latest_results)

@app.route('/model')
def model_page():
    return render_template('model.html')

@app.route('/impact')
def impact_page():
    return render_template('impact.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/api/run-optimization', methods=['POST'])
def run_optimization():
    try:
        data = request.json
        
        # Extract parameters
        num_tasks = data.get('num_tasks', 50)
        num_vms = data.get('num_vms', 10)
        algorithm = data.get('algorithm', 'hybrid')
        
        # Validate inputs
        if num_tasks < 10 or num_tasks > len(tasks_df):
            return jsonify({
                'success': False,
                'message': f'Number of tasks must be between 10 and {len(tasks_df)}'
            }), 400
            
        if num_vms < 5 or num_vms > len(vms_df):
            return jsonify({
                'success': False,
                'message': f'Number of VMs must be between 5 and {len(vms_df)}'
            }), 400
        
        # Subset data
        tasks_subset = tasks_df.head(num_tasks).copy()
        vms_subset = vms_df.head(num_vms).copy()

        # Extract arrays
        task_cpu = tasks_subset['CPU_Req'].values
        task_mem = tasks_subset['Mem_Req'].values
        task_duration = tasks_subset['Duration'].values
        task_priority = tasks_subset['Priority_Encoded'].values
        task_arrival = tasks_subset['Arrival_Time'].values
        vm_cpu = vms_subset['CPU_Capacity'].values
        vm_mem = vms_subset['Mem_Capacity'].values
        vm_energy = vms_subset['Energy_Cost'].values

        # Set global variables in model
        model.task_cpu = task_cpu
        model.task_mem = task_mem
        model.task_duration = task_duration
        model.task_priority = task_priority
        model.task_arrival = task_arrival
        model.vm_cpu = vm_cpu
        model.vm_mem = vm_mem
        model.vm_energy = vm_energy
        model.n_tasks = num_tasks
        model.n_vms = num_vms

        # Run selected algorithm
        np.random.seed(42)

        if algorithm == 'firefly':
            optimizer = FireflyAlgorithmFast(
                n_fireflies=30,
                max_iterations=80,
                alpha=0.2,
                beta0=1.5,
                gamma=0.005,
                n_tasks=num_tasks,
                n_vms=num_vms
            )
            solution, fitness, curve = optimizer.optimize(print_progress=False)

        elif algorithm == 'antlion':
            optimizer = AntLionOptimizerFast(
                n_ants=30,
                max_iterations=80,
                n_tasks=num_tasks,
                n_vms=num_vms
            )
            solution, fitness, curve = optimizer.optimize(print_progress=False)

        elif algorithm == 'hybrid':
            optimizer = OptimizedHybridFast(
                n_agents=30,
                max_iterations=80,
                alpha=0.3,
                beta0=1.5,
                gamma=0.005,
                n_tasks=num_tasks,
                n_vms=num_vms
            )
            # Removed positional arguments as they are set globally in 'model'
            solution, fitness, curve = optimizer.optimize(print_progress=False)

        else:
            return jsonify({
                'success': False,
                'message': 'Invalid algorithm selected'
            }), 400

        # Calculate metrics
        metrics = calculate_metrics(
            solution, algorithm, task_cpu, task_mem, task_duration, task_arrival,
            vm_cpu, vm_mem, vm_energy, num_vms
        )
        
        # Store results
        latest_results[algorithm] = {
            'metrics': metrics,
            'fitness': fitness,
            'convergence': curve
        }
        
        # Return results
        return jsonify({
            'success': True,
            'results': {
                'response_time': metrics['Response_Time'],
                'energy_cost': metrics['Energy_Cost'],
                'makespan': metrics['Makespan'],
                'resource_utilization': metrics['Resource_Utilization'],
                'throughput': metrics['Throughput'],
                'sla_compliance': metrics['SLA_Compliance'],
                'load_balance_score': metrics['Load_Balance_Score'],
                'success_rate': metrics['Success_Rate'],
                'fitness': fitness,
                'algorithm': algorithm
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/get-comparison', methods=['GET'])
def get_comparison():
    try:
        # Get parameters from query string
        num_tasks = int(request.args.get('num_tasks', 50))
        num_vms = int(request.args.get('num_vms', 10))

        # Validate inputs
        if num_tasks < 10 or num_tasks > len(tasks_df):
            return jsonify({
                'success': False,
                'message': f'Number of tasks must be between 10 and {len(tasks_df)}'
            }), 400

        if num_vms < 5 or num_vms > len(vms_df):
            return jsonify({
                'success': False,
                'message': f'Number of VMs must be between 5 and {len(vms_df)}'
            }), 400

        tasks_subset = tasks_df.head(num_tasks).copy()
        vms_subset = vms_df.head(num_vms).copy()

        # Extract arrays
        task_cpu = tasks_subset['CPU_Req'].values
        task_mem = tasks_subset['Mem_Req'].values
        task_duration = tasks_subset['Duration'].values
        task_priority = tasks_subset['Priority_Encoded'].values
        task_arrival = tasks_subset['Arrival_Time'].values
        vm_cpu = vms_subset['CPU_Capacity'].values
        vm_mem = vms_subset['Mem_Capacity'].values
        vm_energy = vms_subset['Energy_Cost'].values

        # Set global variables in model
        model.task_cpu = task_cpu
        model.task_mem = task_mem
        model.task_duration = task_duration
        model.task_priority = task_priority
        model.task_arrival = task_arrival
        model.vm_cpu = vm_cpu
        model.vm_mem = vm_mem
        model.vm_energy = vm_energy
        model.n_tasks = num_tasks
        model.n_vms = num_vms

        results = {}

        # Firefly
        np.random.seed(42)
        fa = FireflyAlgorithmFast(
            n_fireflies=30,
            max_iterations=80,
            alpha=0.2,
            beta0=1.5,
            gamma=0.005,
            n_tasks=num_tasks,
            n_vms=num_vms
        )
        fa_solution, fa_fitness, fa_curve = fa.optimize(print_progress=False)
        results['firefly'] = calculate_metrics(
            fa_solution, 'Firefly', task_cpu, task_mem, task_duration, task_arrival,
            vm_cpu, vm_mem, vm_energy, num_vms
        )

        # AntLion
        np.random.seed(42)
        alo = AntLionOptimizerFast(
            n_ants=30,
            max_iterations=80,
            n_tasks=num_tasks,
            n_vms=num_vms
        )
        alo_solution, alo_fitness, alo_curve = alo.optimize(print_progress=False)
        results['antlion'] = calculate_metrics(
            alo_solution, 'AntLion', task_cpu, task_mem, task_duration, task_arrival,
            vm_cpu, vm_mem, vm_energy, num_vms
        )

        # Hybrid
        np.random.seed(42)
        hybrid = OptimizedHybridFast(
            n_agents=30,
            max_iterations=80,
            alpha=0.3,
            beta0=1.5,
            gamma=0.005,
            n_tasks=num_tasks,
            n_vms=num_vms
        )
        hybrid_solution, hybrid_fitness, hybrid_curve = hybrid.optimize(print_progress=False)
        results['hybrid'] = calculate_metrics(
            hybrid_solution, 'Hybrid', task_cpu, task_mem, task_duration, task_arrival,
            vm_cpu, vm_mem, vm_energy, num_vms
        )

        # Store convergence curves
        results['convergence'] = {
            'firefly': fa_curve,
            'antlion': alo_curve,
            'hybrid': hybrid_curve
        }

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/run-full-optimization', methods=['POST'])
def run_full_optimization_endpoint():
    try:
        data = request.json

        # Extract parameters
        num_tasks = data.get('num_tasks', 1000)
        num_vms = data.get('num_vms', 220)

        # Validate inputs
        if num_tasks < 10 or num_tasks > len(tasks_df):
            return jsonify({
                'success': False,
                'message': f'Number of tasks must be between 10 and {len(tasks_df)}'
            }), 400

        if num_vms < 5 or num_vms > len(vms_df):
            return jsonify({
                'success': False,
                'message': f'Number of VMs must be between 5 and {len(vms_df)}'
            }), 400

        # Run the full optimization
        results = run_full_optimization(num_tasks=num_tasks, num_vms=num_vms, save_results=True)

        # Return results
        return jsonify({
            'success': True,
            'results': {
                'results_df': results['results_df'].to_dict(),
                'convergence_curves': results['convergence_curves'],
                'solutions': results['solutions'],
                'metrics': results['metrics'],
                'diagnostics': results['diagnostics']
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)