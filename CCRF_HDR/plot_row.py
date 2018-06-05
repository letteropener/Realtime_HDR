import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
comp=np.loadtxt("compsum_R_raw.txt",dtype=np.float32)
row=comp[100]

total=np.sum(row)
quarter=int(total/4.0)
count=0
low_index=0
while count<=quarter:
    count+=row[low_index]
    low_index+=1
count=0
high_index=255
while count<= quarter:
    count+=row[high_index]
    high_index-=1
sigma=(high_index-low_index+1)/1.349
'''count = 0
while count<= quarter:
    count+=row[high_index]
    high_index-=1
sigma=(high_index-low_index)/1.349'''

#get mu
weighted_sum=0
i=low_index
count=0
while i<high_index:
    weighted_sum+=i*row[i]
    count+=row[i]
    i+=1

mu=weighted_sum*1.0/count
print(mu,sigma)
x=np.array(range(256))

plt.plot(x,mlab.normpdf(x, mu, sigma),'b',label="normal")
plt.plot(row/total,'r-',label="actual data")
plt.legend()
plt.savefig("plot.jpg")

