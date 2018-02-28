import numpy as np #mathematical operations
import matplotlib.pyplot as plt #nice graphs
from mpl_toolkits.mplot3d import Axes3D #nice 3D graphs

observations = 1000 #This choice affect the speed of the algorithm
#f(x,y) = a*x + b*z + c
#np.random.uniform(low,high,size)
xs = np.random.uniform(low=0,high=10,size=(observations,1)) #observation * 1
zs = np.random.uniform(-10,10,(observations,1))

#inputs From the linear mode
#inputs = n x k = 1000 x 2 the matrixe 2x2
inputs = np.column_stack((xs,zs)) #matrix 1000 by 2
print (inputs.shape) #(1000,2) correct
noise = np.random.uniform(-1,1,(observations,1))
targets =   2*xs   -  3*zs   +   5 +       noise
#1000*1    1000*1    1000*1      scalar    1000*1
print(targets.shape)

targets = targets.reshape(observations,)

#Declare the figure
fig = plt.figure()

#A method allowing us to create the 3D plot
ax = fig.add_subplot(111,projection='3d')

#Choose de axes
ax.plot(xs,zs,targets)

#Set lables
ax.set_xlabel('xs')
ax.set_ylabel('zs')
ax.set_zlabel('Targets')

#You can fiddle with the azim parameter to plot the data
#from different angles. Just change the value of azim=100
#to azim=0; azim=200 or whaterver
ax.view_init(azim=100)
plt.show()
targets = targets.reshape(observations,1)

init_range = 0.1
#our initial weight and biases will be picket randomly from the interval
#[-0.1,0.1]
weights = np.random.uniform(-init_range,init_range,size=(2,1)) #matriz 2x1
biases = np.random.uniform(-init_range,init_range,size=1) #scalar 1
#one biases for one input

print(weights)
print(biases)

learning_rate = 0.02 #this works for this example
for i in range(140):
    outputs = np.dot(inputs,weights) + biases
    deltas = outputs - targets
    loss = np.sum(deltas ** 2) / 2 / observations
    print(loss) #if it is decreasing our ml algorithm functions well
    deltas_scaled = deltas / observations
    weights = weights - learning_rate * np.dot(inputs.T,deltas_scaled)
    #2x1        2x1         scalar              1000x2     1000x1  = 2x1
    #dimensionality check
    #print(weights.shape,inputs.shape,deltas_scaled.shape)
    biases = biases - learning_rate * np.sum(deltas_scaled)
    
plt.plot(outputs,targets)
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()