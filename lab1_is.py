import numpy as np

#Costants
p = 11
M = np.array([[2, 5], 
              [1, 7]])
inv_M = np.linalg.inv(M)    
n = 5

#Funtctions
def transposition(y_i):
    y_i[4:8] = np.flip(y_i[4:8])
    return y_i
    
def linear(Z_i):
    Z_i = Z_i.reshape(2, 4)
    W_i = np.dot(M, Z_i)
    W_i = np.mod(W_i.flatten(), p * np.ones(8))
    return W_i

def inv_linear(w_i):
	W_i = w_i.reshape(2, 4)
	return np.dot(np.linalg.inv(M), W_i).flatten()

def subkey_gen(key):
    k1 = np.array([key[0],key[2],key[4],key[6]])
    k2 = np.array([key[0],key[1],key[2],key[3]])
    k3 = np.array([key[0],key[3],key[4],key[7]])
    k4 = np.array([key[0],key[3],key[5],key[6]])
    k5 = np.array([key[0],key[2],key[5],key[7]])
    k6 = np.array([key[2],key[3],key[4],key[5]])
    return (k1,k2,k3,k4,k5,k6)

def substitution_1(v):
    return np.mod(2 * v, p * np.ones(8))

def inv_substitution_1(y_i):
	arr = y_i / 2
	return arr.astype(int) 

def subkey_sum(z_i, key):
	return np.mod(z_i + np.append(key, key), p * np.ones(8))

def inv_subkey_sum(x, key):
	key_appended = np.append(key, key)
	return x - key_appended

#Tasks
def task_1(n, z_i, w_i, key):
	key_set = subkey_gen(key)
	for i in range(n):
		k_i = key_set[i]
		v_i = np.mod(w_i + np.append(k_i, k_i), p * np.ones(8)) 
		print(v_i) 
		y_i = substitution_1(v_i)
		z_i = transposition(y_i)
		if i < 4:
			w_i = linear(z_i)
	x = subkey_sum(z_i, key_set[5])
	print(x)
	return x

def task_2(x, key):
	key_set = subkey_gen(key)
	z_i = inv_subkey_sum(x, key_set[5])
	y_i = transposition(z_i)
	v_i = inv_substitution_1(y_i)
	for i in range(n - 1, 0, -1):
		w_i = inv_subkey_sum(v_i, key_set[i])
		z_i = inv_linear(w_i)
		y_i = transposition(z_i)
		v_i = inv_substitution_1(y_i)
		print(v_i)
	w_0 = inv_subkey_sum(v_i, key_set[0])
	print(w_0)

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
	z_i = np.array(8)
	w_i = np.array([1, 0, 0, 0, 0, 0, 0, 0])
	key = np.array([1, 0, 0, 0, 0, 0, 0, 0])
	x = task_1(n, z_i, w_i, key)
	task_2(x, key)

if __name__ == "__main__":
	main()
