# Sinh viên không được import thêm thư viện nào khác ngoài thư viện numpy
import numpy as np

MSSV = "31231027916"  # Thay '123456789' bằng mã số sinh viên của bạn


# Sinh viên có thể tự định nghĩa các hàm bổ trợ ở đây



#Chuẩn hoá các giá trị đầu vào
def standardization(P1, S1,P2,S2):
    if P1 < 1 or P1 > 99:
        P1 = max(1, min(P1, 99))
    if S1 < 1 or S1 > 88:
        S1 = max(1,min(S1,88))
    if P2 < 1 or P2 > 99:
        P2 = max(1,min(P2,99))
    if S2 < 1 or S2 > 88:
        S2 = max(1,min(S2,88))
    return P1, S1, P2, S2


#Hàm dành cho trường hợp cơ bản
def base_case (P1,S1,P2,S2,D):
    
    
    if  D <= 5 :
            P1 = P1 * 2
            S1 = S1 // 2
    if D >= 7 and D <= 11:
        P2 = P2 *5
        S2 -= 20
    if D == 12:
        P1 = 99
        S1 = 88
        P2 -= 12
        S2 -= 12
    P1,S1,P2,S2 = standardization(P1,S1,P2,S2)
    result = max((P1 - S2)*(13-D)/999, (S1- P2)*(13-D)/888)
    return result


#Hàm kiểm tra số chính phương
def is_square(x):
    r = int(np.sqrt(x))
    return r*r ==x


#Hàm kiểm tra lập phương
def is_cube(x):
    r = round(x**(1/3))
    return r**3 == x


#Phương pháp chia đôi
def find_root_bisection (f_func, a, b, D):
    n=0
    a_n=a
    b_n=b
    while n < D: 
        x_nB = (a_n+b_n)/2            
        if f_func(a_n) * f_func(x_nB) < 0:
            a_n_1 = a_n
            b_n_1 = x_nB
        else:
            a_n_1 = x_nB
            b_n_1 = b_n
        n+=1
        a_n = a_n_1
        b_n = b_n_1
    return x_nB


#Phương pháp Newton Raphson
def find_root_Newton_Raphson(f_func,f_prime,x_0,D):
    n = 0
    x_nN = x_0
    while n < D:
        x_n_1 = x_nN - f_func(x_nN)/f_prime(x_nN)
        n = n + 1
        x_nN = x_n_1
    return x_nN


#Kiểm tra cấp số cộng
def check_arith(a,b,c,d):
    arr = np.sort([a,b,c,d])
    if arr[0] - arr[1] == arr[1] - arr[2] == arr[2] - arr[3] != 0:
        return True
    return False
    

#Kiểm tra cấp số nhân:
def check_geom(a,b,c,d):
    arr = np.sort([a,b,c,d])
    if arr[0]/arr[1] == arr[1]/arr[2] == arr[2]/arr[3]:
        return True
    return False
    

#Kiểm tra số nguyên tố:
def is_prime(n):
    if n < 2:
        return False
    if n in (2,3):
        return True
    if n % 2 == 0 or n %  3 == 0:
        return False
    i = 5
    while i*i <= n:
        if n % i == 0 or n%(i+2) == 0:
            return False
        i += 6
    return True


#Kiểm tra dãy Fibonacci
def is_fibonacci(a,b,c,d):
    nums = [a,b,c,d]
    nums.sort()
    return nums[2] == nums[0] + nums[1] and nums[3] == nums[1] + nums[2]
#Kiểm tra chéo trội:
def create_matrix(a,b,c,d):

    return np.array([
        [a,34,27,45],
        [21,b,31,37],
        [12,5,c,42],
        [8,19,33,d]
    ])
#Kiểm tra chéo trội
def check_diag_dominance(mat):
    row_total = []
    for i, row in enumerate(mat):
        diag = abs(mat[i][i])
        s = sum(abs(val) for j, val in enumerate(row) if j != i)
        if diag <= s:
            return False
    return True
#Kiểm tra hội tụ bằng bán kính phổ
def spectral_radius(C):
    sr = np.max(np.abs(np.linalg.eigvals(C)))
    if sr < 1:
        return True
    else:
        return False
#Tách ma trận C của Jacobi
def matrix_C_Jacobi(A):
    D = np.diag(np.diag(A))
    L = np.tril(A)-D
    U = np.triu(A)-D
    
    D_inv = np.linalg.inv(D)
    C = -np.dot(D_inv,(L+U))
    return C
#Tách ma trận C của Gauss-Seidel
def matrix_C_GaussSeidel(A):
    D = np.diag(np.diag(A))
    L = np.tril(A)- D
    U = np.triu(A)- D

    DL_inv = np.linalg.inv(D+L)
    C = -np.dot(DL_inv,U)
    return C
#Phương pháp Jacobi
def approximate_solution_Jacobi_method(A,b,x_0,MAX_ITERATIONS):
    D = np.diag(np.diag(A))
    L = np.tril(A)-D
    U = np.triu(A)-D
    
    D_inv = np.linalg.inv(D)
    C = -np.dot(D_inv,(L+U))
    d = np.dot(D_inv,b)
    
    k = 0
    x_k = x_0
    while k < MAX_ITERATIONS:
        x_k_1 = np.dot(C,x_k) + d
        k += 1
        x_k = np.copy(x_k_1)
    return x_k
#Phương pháp Gauss_Seidel
def approximate_solution_GausSeidel_method(A,b,x_0,MAX_ITERATIONS):
    D = np.diag(np.diag(A))
    L = np.tril(A)- D
    U = np.triu(A)- D

    DL_inv = np.linalg.inv(D+L)
    C = -np.dot(DL_inv,U)
    d = np.dot(DL_inv,b)

    k = 0
    x_k = x_0
    while k < MAX_ITERATIONS:
        x_k_1 = np.dot(C,x_k) + d
        k += 1
        x_k = np.copy(x_k_1)
    return x_k
#Nội suy Lagrange
def Lagrange_interpolation(x_values, y_values,x_star):
    def Lagrange_coefficient(k,x):
        result = 1
        for i in np.arange(0,len(x_values),1):
            if i != k:
                result = result*((x-x_values[i])/(x_values[k]-x_values[i]))
        return result
    y_star = 0
    for k in np.arange(0,len(x_values),1):
        y_star = y_star +y_values[k]*Lagrange_coefficient(k,x_star)
    return y_star
#Nội suy Newton




# Sinh viên không được thay đổi định nghĩa hàm calculate và các nội dung có sẵn
#P1,S1 alpha; P2,S2 beta

#Bắt đầu tính toán:
def calculate(P1, S1, P2, S2, D):
    results=[]
    student_out = 0.0
    matched = False 
    
    #Chuẩn hoá
    P1,S1,P2,S2 = standardization(P1, S1,P2,S2)

    #Trường hợp a
    
    if P1 == 99 and S1 == 88:
        results.append(1)
        matched = True
    if P2 == 99 and S2 == 88:
        results.append(0)
        matched = True
    if P1 == P2 == 99 and S1 == S2 == 88 and D != 12:
        results.append(base_case(P1,S1,P2,S2))
        matched = True



    #Trường hợp b
    if D % 2 == 0:
        if P1 > S2 + 20 or S1 > P2 + 20:
            results.append(1 - 0.001*D)
            matched = True


    #Trường hợp c
    if (P1 % 10) % 2 == 0 and (P1 // 10) % 2 == 0 and (S1 % 10) % 2 == 1 and (S1 // 10) % 2 == 1:
        if D <= 6:
            results.append (max((P1 - S2)*(13-D)/999, 
                                (S1- P2)*(13-D)/888) + 0.5)
            matched = True
        elif D >= 7: 
            results.append(max((P1 - S2)*(13-D)/999, 
                               (S1- P2)*(13-D)/888) - 0.2)
            matched = True
            

    #Trường hợp d:
    if (is_square(P1) or is_square(P2)) and (is_cube(S1) or is_cube(S2)):
        results.append(1 - min((P1-S2)*(13-D)/999,
                               (S1-P2)*(13-D)/888))
        matched = True
        

    #Trường hợp e:
    if P1 % S2 == 0 and S1 % P2 == 0:
        def f_func(x):
            return x**4+ 3*x**2 - 2
        def f_prime(x):
            return 4*x**3 + 6*x
        a = 0.2
        b = 1.2
        x_0 = 0.5 +0.01*P1
        x = 0.749368275822
        x_nB = find_root_bisection (f_func, a, b, D)
        x_nN = find_root_Newton_Raphson (f_func,f_prime,x_0,D)
        results.append(np.abs(x_nB-x) - np.abs(x_nN-x))
        matched = True


    #Trường hợp f
 
    if check_geom(P1,P2,S1,S2):
        mat_A = create_matrix(P1,S1,P2,S2)
        b = np.array([[1],[2],[3],[4]])
        x0_J = np.array([[1],[1],[1],[0.01*(P1+S1)]])
        x0_G = np.array([[1],[1],[1],[0.01*(P2+S2)]])

        if check_diag_dominance(mat_A):
            x_j = approximate_solution_Jacobi_method(mat_A,b,x0_J,D)
            x_g = approximate_solution_GausSeidel_method(mat_A,b,x0_G,D )
            results.append(np.linalg.norm(x_j,2)/np.linalg.norm(x_g,2))
            matched = True
        else:
            CJ = matrix_C_Jacobi(mat_A)
            CGS = matrix_C_GaussSeidel(mat_A)
            if spectral_radius(CJ) and not spectral_radius(CGS):
                results.append(1)
                matched = True
            elif spectral_radius(CGS) and not spectral_radius(CJ):
                results.append(0)
                matched = True
            elif not spectral_radius(CJ) and not spectral_radius(CGS):
                results.append(0.5)
                matched = True
            elif  spectral_radius(CGS) and spectral_radius(CJ):
                x_j = approximate_solution_Jacobi_method(mat_A,b,x0_J,D)
                x_g = approximate_solution_GausSeidel_method(mat_A,b,x0_G,D )
                results.append(np.linalg.norm(x_j,2)/np.linalg.norm(x_g,2))
                matched = True


    # #Trường hợp g
    # if is_prime(P2) and is_prime(S2):
    #     results.append(0) #Học xong code tiếp


    # #Trường hợp h
    # if is_fibonacci(P1,S1,P2,S2):
    #     results.append(0)#học xong code tiếp


    # #Trường hợp i
    # if check_arith(P1,S1,P2,S2):
    #     results.append(0)#Học xong code tiếp

        
      #Trường hợp cơ bản:
    if not matched:
        value = base_case(P1,S1,P2,S2,D)
        results.append(value)
        

    
    
    student_out = sum(results)/len(results)
    if student_out < 0 or student_out >1:
        student_out = max(0,min(1,student_out))
    return student_out
P1,S1,P2,S2,D = 41,29,62,12,7
print(calculate(P1,S1,P2,S2,D))

 
