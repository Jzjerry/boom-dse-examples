# Lower-Bound Multi-Objective Max-value Entropy Search from Botorch 

# Example Results

## Using `optimize_acqf` from BoTorch
| Hypervolume | Cost | Total Samples | UUID | 
| ---- | ----  | ---- | ---- |
| 6.2516200710351715 | 1013713.5857478213 | 50 | `00ef538e88634ddd9810d034b748c24d` |
| 6.485904749085943  | 999418.0176275081  | 50 | `c0ffeec0ffeec0ffeec0ffeec0ffeeee` |

## Using direct samples on possible design points
| Hypervolume | Cost | Total Samples | UUID | 
| ---- | ----  | ---- | ---- |
| 6.536375265364496 | 1056758.6832421238 | 50 | `00ef538e88634ddd9810d034b748c24d` |
| 6.490075415303281 | 1010746.8153362505 | 50 | `c0ffeec0ffeec0ffeec0ffeec0ffeeee` |

# Reference

Z. Wang and S. Jegelka, Max-value Entropy Search for Efficient Bayesian Optimization, ICML, 2017.

https://botorch.org/tutorials/information_theoretic_acquisition_functions