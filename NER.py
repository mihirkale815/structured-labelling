import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import anago1
from anago1.reader import load_data_and_labels


'''

f = open('flight_test.bio')
f1 = open('flight_test.txt','w')

Lines=f.readlines()

i=0

for line in Lines:
    try:
         a=line.split()
    
         f1.write(a[1]+'\t'+ a[0]+'\n')
    except:
        f1.write(line)

    
'''


x_train, y_train ,adj_train = load_data_and_labels('ssl_train.txt')
x_valid, y_valid ,adj_valid = load_data_and_labels('ssl_test.txt')
x_test, y_test,adj_test = load_data_and_labels('ssl_test.txt')

model = anago1.Sequence()
model.train(x_train,adj_train, y_train,x_valid, adj_valid,y_valid)
model.eval(x_test,adj_test, y_test)
