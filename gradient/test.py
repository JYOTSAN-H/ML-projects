import numpy as np
import pandas as pd
df=pd.read_csv('test_scores.csv')

x=df.math
y=df.cs
xs=np.array(x)
ys=np.array(y)
def gradient(x,y):
    m_cuurent=b_current=0
    iteration=100
    n=len(x)
    lr=0.0001
    for i in range(iteration):
        
        y_p = m_cuurent * x + b_current
        cost=(1/n) * sum([val**2 for val in (y-y_p)])
        md= -(2/n) * sum(x*(y - y_p))
        bd= -(2/n) * sum(y - y_p)
        m_cuurent=m_cuurent-lr*md
        b_current=b_current-lr*bd
        print("M current="+str(m_cuurent)+"    b current="+str(b_current)+"    iteration=" + str(i)+"  cost= "+str(cost))
gradient(xs,ys)

