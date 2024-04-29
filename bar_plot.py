import numpy as np 
import matplotlib.pyplot as plt 


# number of attributes in plot
n=3

# weights to adjust scale of projected data
w1 = 10
w2 = 1000

data1 = [0.03991749292327016*w1, 0.004176974713981476*w2, -1.2895616094629612] 
data2 = [0.0405137321187725*w1, 0.0030828332373353715*w2, -1.0957333141337418] 
data3 = [0.052122287123348175*w1, 0.0021839245579274983*w2, -1.0076182449188336] 
  
r = np.arange(n) 
width = 0.33
  
# add items in plot
plt.bar(r, data1, color = 'tab:blue', 
        width = width, edgecolor = 'black', 
        label='1*10^-6') 
plt.bar(r + width, data2, color = 'tab:orange', 
        width = width, edgecolor = 'black', 
        label='1*10^-5') 
plt.bar(r + 2 * width, data3, color = 'tab:purple', 
        width = width, edgecolor = 'black', 
        label='1*10^-4') 
   
# add title, labels, legend and ticks

plt.title("Tau") 
# plt.ylabel("Number of people voted")   
# plt.grid(linestyle='--') 
plt.xticks(r + width/2,['T_max (s) * {}'.format(w1),'E_tot (J) * {}'.format(w2),'Reward']) 
plt.legend() 
  
plt.show() 