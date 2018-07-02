
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

    for i in range(length):
        if i<N:
            ave[i] = sum(vector[:2*i+1])/(2*i+1)
        elif i >= length - N:
            ave[i] = sum(vector[i-(length-i-1):])/(2*(length-i-1)+1)
        else:
            ave[i] = sum(vector[i-N:i+N+1])/(2*N+1)
    return ave

if __name__ == "__main__":
    A = [1+1j, 2+2j, 3+3j, 4+4j, 5+5j, 5+5j, 6+6j, 7+7j, 8+8j]
    B = meanFilter(A, 10)
    print(B)



