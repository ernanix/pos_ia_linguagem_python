import pandas as pd
from pandas import Series, DataFrame
import numpy as np

obj = pd.Series([4,-2,1,-9])
obj2 = pd.Series([9,7,5,10], index=['João','Maria','Pedro','José'])


sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)


data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
'year': [2000, 2001, 2002, 2001, 2002, 2003],
'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data,columns=['state','year','pop','debt'])




frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'),
index=['Utah', 'Ohio', 'Texas', 'Oregon'])


"""
import numpy as np
import matplotlib.pyplot as plt

#my_array =  np.arange(1000);


#data = np.random.randn(2,2,2,2)

#zeros = np.zeros((10,10))

arr2d = np.array([[11,12,13],[21,22,23],[31,32,33]])

nsteps = 20
draws = np.random.randint(0,2,size=nsteps)
steps = np.where(draws > 0,1,-1)
walk = steps.cumsum()

fig, ax = plt.subplots()
ax.plot(range(nsteps), np.asarray(walk))
"""

