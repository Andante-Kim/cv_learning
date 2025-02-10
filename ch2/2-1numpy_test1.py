import numpy as np          # numpy 모듈 import, np로 사용

a = np.zeros((2,5), np.int32)   # 초기값이 0인 배열
b = np.ones((3,1), np.uint8)    # 초기값이 1인 배열
c = np.empty((1,5), np.float64) # 초기값이 없는(임의의 값 저장) 배열
d = np.full(5, 15, np.float32)  # 특정 값으로 채워진 배열

print(type(a), type(a[0]), type(a[0][0]))       # 객체 자료형(type) 출력
print(type(b), type(b[0]), type(b[0][0]))           
print(type(c), type(c[0]), type(c[0][0]))
print(type(d), type(d[0]) )

print('c 형태:', c.shape, '   d 형태:', d.shape)    # 객체 형태(shape) 출력

print(a)    # 객체 원소 출력
print(b)
print(c)
print(d)