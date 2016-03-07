from Kernel_SVM_Multiclass import *


plt.close('all')
# New dataset
n = 200
# radius
r1 = 8.0
r2 = 5.0
r3 = 12.0
angles = 2*pi*np.random.rand(n,1) - pi
X1 = np.concatenate([r1*np.cos(angles),r1*np.sin(angles)] , axis=1) + 0.5*np.random.randn(n,2)
X2 = np.concatenate([r2*np.cos(angles),r2*np.sin(angles)] , axis=1) + 0.5*np.random.randn(n,2)
X3 = np.concatenate([r3*np.cos(angles),r3*np.sin(angles)] , axis=1) + 0.5*np.random.randn(n,2)
X = np.concatenate([X1,X2,X3],axis=0)
y = np.ones(3*n)
y[n:(2*n-1)] = 2
y[2*n:] = 3

# Visu
plt.figure()
plt.title('data')
plt.plot(X[:n,0],X[:n,1],'bo')
plt.plot(X[n:(2*n-1),0],X[n:(2*n-1),1],'ro')
plt.plot(X[2*n:,0], X[2*n:,1],'go')
plt.show()

# Prendre un example sur deux pour le test
X_train = X[::2]
X_test = X[1::2]
y_train = y[::2]
y_test = y[1::2]

K_train = pol_kernel(X_train,X_train)
K_test = pol_kernel(X_test,X_train)

# Train SVM
labels, labels_index, alphas, bs, y_oaa = one_vs_all_train(K_train,y_train)

# Predict SVM
y_pred = one_vs_all_predict(K_test, labels, labels_index, alphas, bs, y_oaa)

#Test accuracy
print "Accuracy score: ", me.accuracy_score(y_test,y_pred) 
