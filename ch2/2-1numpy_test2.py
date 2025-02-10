import numpy as np

arr1 =np.arange(10).reshape(2,5)	# 연속적인 숫자 배열 
print(arr1)

print(arr1[0,2]) 	# 원소 색인
print(arr1[0][2])

print(arr1[[0,1],[1,3]]) # [0,1] [1,3] 색인

print(arr1[0])  	# 행 색인
print(arr1[0,])
print(arr1[0,:])

print(arr1[:,0])    	# 열 색인

print(arr1[:,0:3])  	# 슬라이스 색인

print(arr1[:2,3:])  	# 슬라이스 색인
