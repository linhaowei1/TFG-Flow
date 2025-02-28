from dataloader.datasets.qm9 import QM9Dataset
import torch
from torch.distributions import Categorical

class DistributionProperty:
    def __init__(self, dataset, properties=['alpha'], num_bins=1000):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties


        for prop in properties:
            if prop != 'structure':
                asyms = dataset.data['atomic_symbols']
                nodes_arr = torch.tensor([len([a for a in asyms[i] if a != 'H']) for i in range(len(asyms))])
                values = torch.tensor(dataset.data[prop])
                self.distributions[prop] = {}
                self._create_prob_dist(nodes_arr=nodes_arr, values=values, distribution=self.distributions[prop])
            else:
                self.distributions[prop] = {}
                for i in range((dataset.max_length)):
                    self.distributions[prop][i+1] = []
                for feat in dataset:
                    n_nodes = feat['num_atoms'].item()
                    self.distributions[prop][n_nodes].append(feat['structure'])

    def _create_prob_dist(self, nodes_arr, values, distribution):
        
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {'probs': probs, 'params': params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins #min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min)/prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = histogram / torch.sum(histogram)
        probs = Categorical(probs)
        params = [prop_min, prop_max]
        return probs, params

    def sample(self, n_nodes=9):
        vals = []
        for prop in self.properties:
            if prop == 'structure':
                idx = torch.randint(0, len(self.distributions[prop][n_nodes]), (1,))
                val = self.distributions[prop][n_nodes][idx]
                vals.append(val)
                continue
            dist = self.distributions[prop][n_nodes]
            idx = dist['probs'].sample((1,))
            val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val

if __name__ == '__main__':

    dataset = QM9Dataset(fingerprint=True)
    dist = DistributionProperty(dataset, properties=['structure', 'mu'], )
    print(dist.sample(9))