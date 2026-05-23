# Exploring-Python-Library-Gymnasium
[Gymnasium](https://gymnasium.farama.org/) is an open-source Python library used for **Reinforcement Learning (RL)**. It provides a standardized interface for agents to interact with various simulation environments, allowing them to learn through trial and error.

## Installation Guide

Follow these steps to set up Gymnasium on your local machine.

### 1. Install Python
Download the latest version from the [official Python website](https://www.python.org/).

> **⚠️ Important during installation:**
> * Check the box **"Add Python to PATH"**.
> * Gymnasium officially supports **Python 3.10 through 3.13**.

To verify your installation, open your terminal (Command Prompt or PowerShell) and run:
```
python --version
```
You should see something like:
```
Python 3.12.x
```

### 2. Create a Virtual Environment to keep your packages isolated for this project
Run:
```
py -m venv .venv
.venv\Scripts\activate
```

If successful, your terminal should now show:
```
(.venv)
```

### 3. Upgrade pip
python -m pip install --upgrade pip

### 4. Install Gymnasium
Run:
```
pip install gymnasium
```

This installs:
Core Gymnasium package
Basic RL environments

### 5. Install Extra Environment Packages
Gymnasium separates some environment families into optional installs.
Atari environments:
```
pip install "gymnasium[atari]"
```

Everything:
```
pip install "gymnasium[all]"
```