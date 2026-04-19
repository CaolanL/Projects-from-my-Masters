import numpy as np

def linear_kernel_evaluation(x, y):
  return np.inner(x, y)

def second_order_kernel_evaluation(x,y,c=0.1):
  return (1 + c*np.inner(x, y))**2

def third_order_kernel_evaluation(x, y, c=0.1):
  return (1+ c*np.inner(x, y))**3

def rbf_kernel_evaluation(x,y,c=0.1):
  return np.exp(-c * np.linalg.norm(x-y)**2)

def laplacian_kernel_evaluation(x,y,c=0.1):
  return np.exp(-c * np.linalg.norm(x-y, ord = 1))



def linear_kernel():
  return lambda x, y: linear_kernel_evaluation(x, y)

def second_order_kernel(c=0.1):
  return lambda x, y: second_order_kernel_evaluation(x, y, c)

def third_order_kernel(c=1):
  return lambda x, y: third_order_kernel_evaluation(x, y, c)

def rbf_kernel(c=0.1):
  return lambda x, y: rbf_kernel_evaluation(x, y, c)

def laplacian_kernel(c=0.1):
  return lambda x, y: laplacian_kernel_evaluation(x, y, c)


