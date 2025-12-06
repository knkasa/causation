# https://medium.com/@brian-curry-research/introducing-pycausalsim-stop-guessing-start-simulating-causality-958793fa655d
# github= https://github.com/Bodhi8/pycausalsim
# Still building as of 2025/12/6

'''
PyCausalSim is a Python framework for causal discovery and 
inference through simulation. Unlike correlation-based approaches,
PyCausalSim uses counterfactual simulation to establish true 
causal relationships from observational data, specifically 
designed for digital metrics optimization.
'''

#------- Ecommerce conversion simulation --------------------------
import pandas as pd
from pycausalsim import CausalSimulator

# Load your data
data = pd.read_csv('conversion_data.csv')

# Initialize simulator
simulator = CausalSimulator(
    data=data,
    target='conversion_rate',
    treatment_vars=['page_load_time', 'price', 'design_variant'],
    confounders=['traffic_source', 'device_type', 'time_of_day']
    )

# Step 1: Discover causal structure
simulator.discover_graph(method='notears')
simulator.plot_graph()  # Visualize learned causal relationships

# Step 2: Simulate interventions
load_time_effect = simulator.simulate_intervention(
    variable='page_load_time',
    value=2.0,  # seconds
    n_simulations=1000
    )

print(load_time_effect.summary())
# Output:
# Intervention: page_load_time = 2.0
# Current value: 3.5
# Effect on conversion_rate: +2.3% (95% CI: [1.8%, 2.8%])
# P-value: 0.001

# Step 3: Rank all causal drivers
drivers = simulator.rank_drivers()
for var, effect in drivers:
    print(f"{var}: {effect:.3f}")

# Step 4: Find optimal policy
optimal = simulator.optimize_policy(
    objective='maximize_conversion',
    constraints={'price': (10, 50)}
)