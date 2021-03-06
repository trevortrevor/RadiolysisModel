
import numpy as np # Import numpy (mathmatical functions  etc.)
from scipy.integrate import solve_ivp # Uses the scipy ODE solver
import matplotlib.pyplot as plt # functions to plot data
import csv #needed to parse the input data files
rdata = {} #initialise a variable to hold the reaction data

# rdata structure
# key = reaction index
# 0 = number of rectants
# 1 = number of products
# 2 = k_opt
# 3 = k_rec
# 4 = Ea
# 5,6,7 = reactants
# 8,9,10 = products

with open("data1.csv", "r") as csvfile: #This loop reads the data1.csv file into the variable initialised above.
    idata = csv.reader(csvfile)
    next(idata, None) # Skip file header
    for row in idata: # Add all reactions listed
        rdata[row[0]] = row[1:]
        
#Parse Reaction Data
rdata_ints = [0,1] #Defines which variables in the rdata are integers
rdata_floats = [2,3,4] #Defines which variables in the rdata are floating point
reactants = set() #Generates a list of all the reactants (no duplicates)
products = set() #Generates a list of all the products (no duplicates)
for i in rdata: #Goes through each line in the rdata variable
    for j in rdata_ints: # Convert int values from CSV
        try: #Need the try / except to handle blank data
            rdata[i][j] = int(rdata[i][j])
        except:
            pass
    for j in rdata_floats: # Convert float values from CSV
        try:
            rdata[i][j] = float(rdata[i][j])
        except:
            pass    
    for j in range (5,8): # Unique set of reactants 
        try:
            reactants.add(rdata[i][j])
        except:
            pass
    for j in range (8,11): # Unique set of products
        try:
            products.add(rdata[i][j])
        except:
            pass
try: #remove blank values
    products.remove('')
except:
    pass
try: #remove blank values
    reactants.remove('')
except:
    pass

if len(products) != len(reactants): # not very clever parity check
    print('Input-output mismatch in reaction data')

def reaction(t, y, k1, k2): #Defines the ODE type function that will actually be solved
    r1 = k1 * y[0] * y[1]
    r2 = k2 * y[2]
    return [r2-r1, r2-r1, r1-r2] #return concentrations at time t.

k_data = (rdata['1'][2], rdata['2'][2])
tout = [0, 10]
y0 = [1, 1, 1]
solution = solve_ivp(reaction, tout, y0, method='BDF', args=k_data, atol=1E-8 )
print(solution.message)
plt.plot(solution.t, solution.y[0])
plt.plot(solution.t, solution.y[1])
plt.plot(solution.t, solution.y[2])
plt.xscale('log')
plt.yscale('log')
_ = plt.legend(['H+', 'OH-', 'H2O'])
plt.show()
