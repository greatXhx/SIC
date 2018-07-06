
class CustomError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self) #初始化父类
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo

def meanFilter(vector, N):
    ##一维向量的2N+1点均值滤波，取前N点+后N点+自身
    ##self: 输入一维向量
    ##N：N点均值滤波,
    from copy import deepcopy
    length = len(vector)
    ave = deepcopy(vector)

    if length < 2*N+1:
        raise CustomError("向量长度必须大于2N*1")

    if N == 0:
        return vector

    for i in range(length):
        if i<N:
            ave[i] = sum(vector[:2*i+1])/(2*i+1)
        elif i >= length - N:
            ave[i] = sum(vector[i-(length-i-1):])/(2*(length-i-1)+1)
        else:
            ave[i] = sum(vector[i-N:i+N+1])/(2*N+1)
    return ave

def complexMedFilter(vector, N):
    ##复数中位数滤波，取前后加本身工N点
    ##self: 输入一维向量
    ##N：N点中位数滤波
    import numpy as np

    length = len(vector)
    vectorNp = np.array(vector, dtype=complex)
    vectorAbs = abs(vectorNp)
    # print(vectorAbs)
    halfN = int((N-1)/2)

    if length < N:
        raise CustomError("向量长度必须大于N")

    if N == 0:
        return vector

    for i in range(length):
        if i < (N-1)/2:
            sortindex = np.argsort(vectorAbs[:2*i+1])
            vector[i] = vectorNp[sortindex[i]]
        elif length-i <= (N-1)/2:
            sortindex = np.argsort(vectorAbs[i-(length-i)+1:])
            vector[i] = vectorNp[i-(length-i)+1 + sortindex[length-i-1]]
        else:
            sortindex = np.argsort(vectorAbs[i-halfN:i+N-halfN])
            vector[i] = vectorNp[i - halfN +sortindex[halfN]]

    return vector



if __name__ == "__main__":

    A = [0+4j, 0+1j, 0+3j, 0+4j, 0+2j, 0+5j, 0+9j, 0+1j, 0+8j]
    # B = meanFilter(A, 0)
    # print(B)
    complexMedFilter(A, 7)


