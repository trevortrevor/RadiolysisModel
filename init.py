
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import csv
rdata = {}

# rdata structure
# key = reaction index
# 0 = number of rectants
# 1 = number of products
# 2 = k_opt
# 3 = k_rec
# 4 = Ea
# 5,6,7 = reactants
# 8,9,10 = products

with open("data1.csv", "r") as csvfile:
    idata = csv.reader(csvfile)
    next(idata, None) # Skip file header
    for row in idata: # Add all reactions listed
        rdata[row[0]] = row[1:]
        
#Parse Reaction Data
rdata_ints = [0,1]
rdata_floats = [2,3,4]
reactants = set()
products = set()
for i in rdata:
    for j in rdata_ints:
        try:
            rdata[i][j] = int(rdata[i][j])
        except:
            pass
    for j in rdata_floats:
        try:
            rdata[i][j] = float(rdata[i][j])
        except:
            pass    
    for j in range (5,8):
        reactants.add(rdata[i][j])
    for j in range (8,11):
        try:
            products.add(rdata[i][j])
        except:
            pass
try:
    products.remove('')
except:
    pass
try:
    reactants.remove('')
except:
    pass

if len(products) != len(reactants):
    print('Input-output mismatch in reaction data')

def reaction(t, y, k1, k2):
    #y[0] = H+, 1 = OH-, 2 = H2O
    r1 = k1 * y[0] * y[1]
    r2 = k2 * y[2]
    return [r2-r1, r2-r1, r1-r2]

k_data = (rdata['1'][2], rdata['2'][2])
tout = [0, 10]
y0 = [1, 1, 1,]
solution = solve_ivp(reaction, tout, y0, method='BDF', args=k_data, atol=1E-8 )
print(solution.message)
plt.plot(solution.t, solution.y[0])
plt.plot(solution.t, solution.y[1])
plt.plot(solution.t, solution.y[2])
plt.xscale('log')
plt.yscale('log')
_ = plt.legend(['H+', 'OH-', 'H2O'])
plt.show()
