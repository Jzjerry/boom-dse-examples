# Lower-Bound Multi-Objective Joint Entropy Search from Botorch 

# Example Results

## Using `optimize_acqf` from BoTorch
| Hypervolume | Cost | Total Samples | UUID | 
| ---- | ----  | ---- | ---- |
| 6.258681678955787 | 1006390.7821321192 | 50 | `00ef538e88634ddd9810d034b748c24d` |
| 6.2646057198812   | 1057176.733044651  | 50 | `c0ffeec0ffeec0ffeec0ffeec0ffeeee` |

## Using direct samples on possible design points
| Hypervolume | Cost | Total Samples | UUID | 
| ---- | ----  | ---- | ---- |
| 6.536375265364496 | 1056803.8231509144 | 50 | `00ef538e88634ddd9810d034b748c24d` |
| 6.490075415303281 | 1010788.7447600595 | 50 | `c0ffeec0ffeec0ffeec0ffeec0ffeeee` |

# Reference

B. Tu, A. Gandy, N. Kantas and B. Shafei, Joint Entropy Search for Multi-Objective Bayesian Optimization, NeurIPS, 2022.

https://botorch.org/tutorials/information_theoretic_acquisition_functions