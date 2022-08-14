import csv
import matplotlib.pyplot as plt
import numpy as np

results1 = []
results2 = []
results3 = []
results4 = []
results5 = []
results6 = []
epoch = []
x = []
y = []
z = []
a = []
b = []
c = []
acc = []

with open('./loss_log/T_log_CNN.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the headers
    for row in reader:
        results1.append(row)
for event in results1:
    epoch.append(event[0])
    x.append(event[1])

with open('./loss_log/T_log_Res.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the headers
    for row in reader:
        results2.append(row)
for event in results2:
    y.append(event[1])

with open('./loss_log/T_log_Auto.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the headers
    for row in reader:
        results3.append(row)
for event in results3:
    z.append(event[1])

with open('./loss_log/T_log_CNNAug.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the headers
    for row in reader:
        results4.append(row)
for event in results4:
    a.append(event[1])

with open('./loss_log/T_log_ResAug.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the headers
    for row in reader:
        results5.append(row)
for event in results5:
    b.append(event[1])

with open('./loss_log/T_log_AutoAug.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the headers
    for row in reader:
        results6.append(row)
for event in results6:
    c.append(event[1])


epoch = [np.float32(i) for i in epoch]
x = [np.float32(i) for i in x]
y = [np.float32(i) for i in y]
z = [np.float32(i) for i in z]
a = [np.float32(i) for i in a]
b = [np.float32(i) for i in b]
c = [np.float32(i) for i in c]

plt.figure(figsize=(8,8))
plt.title('Validation Loss of training different models')
plt.plot(epoch,x)
plt.plot(epoch,y)
plt.plot(epoch,z)
plt.plot(epoch,a)
plt.plot(epoch,b)
plt.plot(epoch,c)
#plt.plot(acc)
plt.legend(['CNN','ResNet','Autoencoder','CNN+Aug','ResNet+Aug','Autoencoder+Aug'])
plt.show()




