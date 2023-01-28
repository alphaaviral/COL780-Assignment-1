import numpy as np
import csv
# sigma = 0
# difference = 0
res = np.zeros((256,300))

for difference in range(0,256):
    for sigma in range(1,300):
        res[difference][sigma-1] = (1.0*(np.exp((np.power(difference,2.0))/(-1*2*(np.power(sigma/100,2.0)))))/(np.power(2*np.pi,0.5)*sigma/100))

with open("lookupTable.csv", "w") as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(res)
