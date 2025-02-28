import os
import json
import numpy as np
logs = './storage/logs'
results = os.listdir(logs)
results = [r for r in results if 'cv' in r]

metrics = {}

for r in results:
    path = os.path.join(logs, r, 'results.json')
    temperature = r.split('temperature=')[1].split('+')[0]
    rho = r.split('rho=')[1].split('+')[0]
    mu = r.split('mu=')[1].split('+')[0]
    
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            if data['validity'] >= 0.75:
                if (temperature, rho, mu) not in metrics:
                    metrics[(temperature, rho, mu)] = []
                metrics[(temperature, rho, mu)].append(data['mae_1'])
    
# get average and std
for key in metrics:
    metrics[key] = (sum(metrics[key]) / len(metrics[key]), np.std(metrics[key]))

# sorted by average
metrics = dict(sorted(metrics.items(), key=lambda item: item[1][0]))
print(metrics)