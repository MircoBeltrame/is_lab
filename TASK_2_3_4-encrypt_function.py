
#the encrypt function takes n the number of rounds, the message and the key to be used. 
#could implement the use of different substitution function passed by parameter


def encrypt(n, message, key,substitution):
  key_set = subkey_gen(key)
  z_i = np.array(8)
  w_i = message
  for i in range(n):
    k_i = key_set[i]
    v_i = np.mod(w_i + np.append(k_i, k_i), p * np.ones(8))  
    y_i = substitution(v_i)
    z_i = transposition(y_i)
    if i < 4:
      w_i = linear(z_i) 
      w_i.flatten()
  x = np.mod(z_i + np.append(key_set[5], key_set[5]), p * np.ones(8))
  return(x)
#---------------------------------------------
#"""##TASK2"""

def inv_linear(w):
  M_inv =np.array([[2,8],[6,10]]) 
  #it's always this, so we can compute it once. computed with sympy
  #in the same way that in task 3 we compute A_inv
  w = w.reshape(2,4)
  z = np.dot(M_inv,w)
  z = np.mod(z,11*np.ones(z.shape))
  return z.flatten()

def inv_substitution_1(y):
  return np.mod(6 * y,p* np.ones(8))

def decrypt(n,x,key,inv_substitution):
  key_set = subkey_gen(key)
  z_i = np.mod(x - np.append(key_set[5], key_set[5]), 11 * np.ones(8))
  y_i = transposition(z_i)
  v_i = inv_substitution(y_i)
  for i in range(4):
    w_i = np.mod(v_i - np.append(key_set[4-i], key_set[4-i]), 11 * np.ones(8))
    z_i = inv_linear(w_i)
    y_i = transposition(z_i)
    v_i = inv_substitution(y_i)
  u = np.mod(v_i - np.append(key_set[0], key_set[0]), 11 * np.ones(8))
  return(u)

def task_2(x, key):
  u = decrypt(5,x,key,inv_substitution_1)
  print(u)
#--------------------------------------------
#"""##TASK 3"""

def task_3():
  A = []
  B = []
  for i in range(8):
    a_i = encrypt(5, np.zeros(8), np.eye(8)[i],substitution_1)
    b_i = encrypt(5, np.eye(8)[i], np.zeros(8),substitution_1)
    A = np.append(A, a_i)
    B = np.append(B, b_i)
  A = A.reshape(8,8).T
  B = B.reshape(8,8).T
  return (A,B)

A,B = task_3()

#------------------------------------------------------------------------
#"""##TASK 4"""

import sympy as sp
# u is the plaintext, x the ciphertext
def task_4(u,x):
  A,B = task_3()  
  A = np.array(A,dtype='int32')
  B = np.array(B,dtype='int32')
   
  A = sp.Matrix(A.tolist())
  B = sp.Matrix(B.tolist())
  x = sp.Matrix(x)
  u = sp.Matrix(u)

  A_inv = A.inv()
  A_det = A.det()
  A_det_inv = sp.invert(A_det,11)
  A_inv = A_inv * A_det_inv * A_det
  K = A_inv*(x-B*u)

  return(K.tolist())

k = task_4([7,2,3,1,8,9,0,8],[6,1,8,2,2,10,10,1])
k = np.array(k,dtype='int32').flatten()
k = np.mod(k,p*np.ones(8))
print(k)

k = task_4([4,6,0,7,6,9,5,9],[0,5,6,10,9,4,4,8])
k = np.array(k,dtype='int32').flatten()
k = np.mod(k,p*np.ones(8))
print(k)
