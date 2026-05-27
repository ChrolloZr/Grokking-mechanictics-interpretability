Timeless: Neural Mechanics & Kinematic Grokking
This repository contains the core research, implementation, and interactive dashboard for investigating parameter crystallization and structural phase transitions in modular deep learning models.

The project explores how models learn to perform modular arithmetic tasks, transitioning from a memorization phase to a "grokking" phase where they discover underlying mathematical Fourier structures.

Project Overview

grokking_code.py: A standalone PyTorch research pipeline. It implements a Transformer-like architecture to solve modular addition tasks, monitors the "grokking" phase transition, and performs a Discrete Fourier Transform (DFT) on the learned embedding weights to visualize structural frequency patterns.

index.html, script.js, & style.css: An interactive web-based dashboard designed to visualize the internal mechanics of these models. Users can explore how weight embeddings crystallize into trigonometric basis functions during training.

Key Research Components

Phenomenon of Grokking and Co-Grokking

Mechanistic Interpretability: Analysis of weight matrix norms to detect sparse structural Fourier components.

Phase Transition Monitoring: Real-time tracking of validation accuracy and loss to identify the exact epoch of grokking.

Kinematic Visualization: Interactive sandbox to observe the "physics" of modular deep learning models.

Usage Prerequisites

Python 3.x

PyTorch

NumPy & Matplotlib


The live research dashboard is hosted via GitHub Pages:
 https://chrollozr.github.io/Grokking-mechanictics-interpretability/

Credits
This project was developed by Moussa-Alioune Taboure at Vanier College
Research assistance provided by mai.trigraph.org and Ivan T. Ivanov
