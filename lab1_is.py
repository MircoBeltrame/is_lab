import numpy as np

#Costant
p = 11
M = np.array([[2, 5], 
              [1, 7]])
inv_M = np.linalg.inv(M)    

#Funtction
def transposition(y_i):
    y_i[4:8] = np.flip(y_i[4:8])
    return y_i

def inv_transposition(y_i):
    y_i[4:8] = np.flip(y_i[4:8])
    return y_i
    
def linear(Z_i):
    Z_i = Z_i.reshape(2, 4)
    W_i = np.dot(M, Z_i)
    W_i = np.mod(W_i.flatten(), p * np.ones(8))
    return W_i

def inv_linear(W_i):
	W_i = W_i + p
	Z_i = np.dot(W_i, inv_M)
	return Z_i.flatten()

def subkey_gen(key):
    k1 = np.array([key[0],key[2],key[4],key[6]])
    k2 = np.array([key[0],key[1],key[2],key[3]])
    k3 = np.array([key[0],key[3],key[4],key[7]])
    k4 = np.array([key[0],key[3],key[5],key[6]])
    k5 = np.array([key[0],key[2],key[5],key[7]])
    k6 = np.array([key[2],key[3],key[4],key[5]])
    return (k1,k2,k3,k4,k5,k6)

def inv_subkey_gen(key):
	return 0

def substitution_1(v):
    return np.mod(2 * v, p * np.ones(8))

def inv_substitution_1(v):
	return v + p / 2

#Tasks
def task_1(n, z_i, w_i, key):
	key_set = subkey_gen(key)
	for i in range(n):
		k_i = key_set[i]
		v_i = np.mod(w_i + np.append(k_i, k_i), p * np.ones(8))  
		y_i = substitution_1(v_i)
		z_i = transposition(y_i)
		if i < 4:
			w_i = linear(z_i)
			w_i.flatten()
	x = np.mod(z_i + np.append(key_set[5], key_set[5]), p * np.ones(8))
	print(x)

def task_2(crypted):
	return 0

def task_3():
	return 0

def task_4():
	return 0

def task_5():
	return 0

def task_6():
	return 0

def task_7():
	return 0

def task_8():
	return 0

#Main
def main():
	n = 5
	z_i = np.array(8)
	w_i = np.array([1, 0, 0, 0, 0, 0, 0, 0])
	key = np.array([1, 0, 0, 0, 0, 0, 0, 0])
	task_1(n, z_i, w_i, key)	

if __name__ == "__main__":
	main()
