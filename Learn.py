import numpy as np
import torch

if __name__=="__main__":
    a = np.random.randint(0,2,(2,1,4,4))
    b = np.random.randint(0,2,(2,1,4,4))

    print(a[0])
    print(a[1])
    print(b[0])
    print(b[1])


    a_ = torch.from_numpy(a)
    b_ = torch.from_numpy(b)

    sum1 = torch.sum(a_, dim=(3,2,1))
    sum2 = torch.sum(b_, dim=(3,2,1))

    print("sum1",sum1)
    print("sum2",sum2)

    denominator = sum1+sum2#分母

    print("分母",denominator)

    mul = torch.mul(a_, b_)
    print("点积",mul)

    sum3 = torch.sum(mul, dim=(3,2,1))
    print("分子和",sum3)

    print(sum3.to(dtype=torch.float)/(sum1+sum2).to(dtype=torch.float))
