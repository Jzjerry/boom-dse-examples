# Lower-Bound Multi-Objective Max-value Entropy Search from Botorch 

# Example Results

## Using `optimize_acqf` from BoTorch

| Hypervolume | Cost | Total Samples(Init,Suggest) | UUID | 
| ---- | ----  | ---- | ---- |
| 6.2516200710351715 | 1013713.5857478213 | 50(30+20) | `00ef538e88634ddd9810d034b748c24d` |
| 6.485904749085943  | 999418.0176275081  | 50(30+20) | `c0ffeec0ffeec0ffeec0ffeec0ffeeee` |

## Using direct samples on possible design points

**Pure Random Initial Sampling**
| Hypervolume | Cost | Total Samples(Init,Suggest) | UUID | 
| ---- | ----  | ---- | ---- |
| 6.062133830345928 | 1038347.5999592122 | 50(30+20) | `00ef538e88634ddd9810d034b748c24d` |

**Latin Hypercube Initial Sampling:**

| Hypervolume | Cost | Total Samples(Init,Suggest) | UUID | 
| ---- | ----  | ---- | ---- |
| 6.536375265364496 | 1056758.6832421238 | 50(30+20) | `00ef538e88634ddd9810d034b748c24d` |

**Sobol Random Initial Sampling:**

| Hypervolume | Cost | Total Samples(Init,Suggest) | UUID | 
| ---- | ----  | ---- | ---- |
| 6.7125230477757505 | 1048965.700307061 | 50(30+20) | `00ef538e88634ddd9810d034b748c24d` |


**MicroAL Initial Sampling:**

| Hypervolume | Cost | Total Samples(Init,Suggest) | UUID | 
| ---- | ----  | ---- | ---- |
| 6.081340689266256 | 1143845.2954964282 | 50(30+20) | `00ef538e88634ddd9810d034b748c24d` |

# Reference

Z. Wang and S. Jegelka, Max-value Entropy Search for Efficient Bayesian Optimization, ICML, 2017.

https://botorch.org/tutorials/information_theoretic_acquisition_functions