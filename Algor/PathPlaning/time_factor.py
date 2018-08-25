import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 分析影响时间的因素

a_factor = np.array([[20088, 386378, 253699, 420046, 498945, 3527373],
                     [1, 21, 14, 23, 29, 209]])
b_factor = np.array([[301, 148420, 12552, 10854, 4880, 461403],
                     [0, 8, 0, 0, 0, 27]])
d_factor = np.array([[323556, 1580475, 253699, 2305326, 2790297, 6959729],
                     [18, 93, 133, 168, 166, 448]])

a_length = np.array([341,1225,2075,1788,1499,5314])
b_length = np.array([341,1244,2153,2145,1658,5895])
d_length = np.array([341,1225,2075,1773,1462,5307])

sns.set()
plt.scatter(a_factor[0], a_factor[1], label='A*')
plt.scatter(d_factor[0], d_factor[1], label='Dijstra')
plt.scatter(b_factor[0], b_factor[1], label='BFS')

z1 = np.polyfit(a_factor[0], a_factor[1], 1)
z2 = np.poly1d(z1)
new_x = np.arange(0,7e6,10000)
new_y = z2(new_x)
print(z2)
plt.plot(new_x,new_y)
plt.xlabel(u'PointNumbers')
plt.ylabel(u'PlanningTime\s')
plt.legend()
dir = sys.path[0]
plt.savefig(os.path.join(dir, 'Results\Time_factor_fit.png'))
plt.show()
