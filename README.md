# Hybrid Firefly–Antlion Optimizer with Energy-Aware Load Balancing for Sustainable Cloud Computing

## 📌 Project Overview

This project proposes a **Hybrid Firefly–Antlion Optimizer (HFAO)** integrated with an **Energy-Aware Load Balancing** strategy to enhance resource utilization, reduce energy consumption, and improve overall performance in cloud computing environments. The hybrid metaheuristic combines the **exploration capability of the Antlion Optimizer (ALO)** with the **exploitation strength of the Firefly Algorithm (FA)** to achieve faster convergence and better optimization results.

The system intelligently schedules and balances workloads across virtual machines (VMs) by considering energy efficiency, response time, and system load, making it suitable for **sustainable and green cloud computing**.

---

## 🎯 Objectives

* Minimize overall **energy consumption** in cloud data centers
* Improve **load balancing efficiency** among virtual machines
* Reduce **task execution time** and **response latency**
* Avoid VM overloading and underutilization
* Enhance **scalability and sustainability** of cloud infrastructure

---

## 🧠 Key Concepts Used

* Cloud Computing & Virtualization
* Metaheuristic Optimization Algorithms
* Firefly Algorithm (FA)
* Antlion Optimizer (ALO)
* Energy-Aware Scheduling
* Load Balancing Techniques

---

## ⚙️ System Architecture

1. **Task Submission** – User tasks are submitted to the cloud
2. **Resource Monitoring** – VM load, energy usage, and capacity are tracked
3. **Hybrid Optimization Engine**

   * Antlion Optimizer handles global search (exploration)
   * Firefly Algorithm refines solutions locally (exploitation)
4. **Energy-Aware Load Balancer** – Assigns tasks to optimal VMs
5. **Performance Evaluation** – Measures energy, response time, and utilization

---

## 🛠️ Technology Stack

* **Programming Language:** Python
* **Simulation Environment:** CloudSim / Custom Cloud Simulator
* **Libraries & Tools:**

  * NumPy – Numerical computations
  * Pandas – Data handling
  * Matplotlib / Seaborn – Performance visualization
  * Scikit-learn (optional) – Performance analysis

---

## 🔄 Hybrid Optimization Workflow

1. Initialize population (tasks and VM allocation)
2. Apply Antlion Optimizer for global search
3. Apply Firefly Algorithm for local refinement
4. Evaluate fitness based on:

   * Energy consumption
   * Load balance factor
   * Execution time
5. Update solutions iteratively
6. Select optimal VM-task mapping

---

## 📊 Performance Metrics

* Total Energy Consumption
* Average Response Time
* Makespan
* VM Utilization Rate
* Load Balance Index


## 📈 Results & Analysis

Experimental results show that the proposed **Hybrid Firefly–Antlion Optimizer** outperforms traditional load balancing techniques by:

* Reducing energy consumption
* Improving VM utilization
* Achieving better load distribution
* Lowering execution time

Graphs and comparative analysis are available in the `results/` directory.

---

## 🚀 Future Enhancements

* Integration with real cloud platforms (AWS, Azure, OpenStack)
* Support for container-based environments (Docker, Kubernetes)
* Incorporation of AI-based workload prediction
* Multi-objective optimization with SLA constraints


## 📜 License

This project is developed for **academic and research purposes**. Feel free to use and modify it with proper attribution.

