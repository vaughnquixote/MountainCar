import numpy as np

a = np.array([1,2,3,4,5])
b = [1,3]

print(a[b])
print(np.sum(a[b]))
print(a[b] - 1)