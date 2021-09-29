
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def rhs(t, y, kf, kb):
    rf = kf * y[0]**2 * y[1]
    rb = kb * y[2]**2
    return [2*(rb - rf), rb - rf, 2*(rf -rb)]

tout = [0, 10]
k_vals = (0.42, 0.17)
y0 = [1, 1, 0,]
solution = solve_ivp(rhs, tout, y0, args=k_vals)

plt.plot(solution.t, solution.y[0])
plt.plot(solution.t, solution.y[1])
plt.plot(solution.t, solution.y[2])
_ = plt.legend(['NO', 'Br$_2$', 'NOBr'])
plt.show()
