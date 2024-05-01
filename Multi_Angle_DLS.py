import matplotlib.pyplot as plt
from scipy import special
from scipy import optimize,linalg
from scipy.optimize import curve_fit
from numpy import exp,pi,sin,cos,argsort,zeros,mean,linspace,sum,matrix,bmat,argmax,transpose,dot,diag
from numpy import diff,insert,array,spacing,eye,sign
from scipy.optimize import nnls
import numpy as np
import csv
from math import sqrt
from tkinter import *
from tkinter.filedialog import *
from PIL import Image,ImageTk
import random

#实验数据文件输入,单角度/双角度/三角度

window = Tk()
window.title("多角度动态光散射粒径分析")
window.geometry('1200x700')

#参数输入界面
nm_tit = Label(window,text='介质折射率：')
nm_tit.place(relx=0.05, rely=0.05)
nm_inp = Entry(window)
nm_inp.place(relx=0.2, rely=0.05)

np_tit = Label(window,text='颗粒折射率：')
np_tit.place(relx=0.05, rely=0.1)
np_inp = Entry(window)
np_inp.place(relx=0.2, rely=0.1)

lm_tit = Label(window,text='入射光波长(nm)：')
lm_tit.place(relx=0.05, rely=0.15)
lm_inp = Entry(window)
lm_inp.place(relx=0.2, rely=0.15)


Dmin_tit = Label(window,text='反演范围下限(nm)：')
Dmin_tit.place(relx=0.05, rely=0.25)
Dmin_inp = Entry(window)
Dmin_inp.place(relx=0.2, rely=0.25)

Dmax_tit = Label(window,text='反演范围上限(nm)：')
Dmax_tit.place(relx=0.05, rely=0.3)
Dmax_inp = Entry(window)
Dmax_inp.place(relx=0.2, rely=0.3)

N_tit = Label(window,text='取样点数：')
N_tit.place(relx=0.05, rely=0.35)
N_inp = Entry(window)
N_inp.place(relx=0.2, rely=0.35)

ann_tit = Label(window,text='散射角个数：')
ann_tit.place(relx=0.05, rely=0.4)
ann_inp = Entry(window)
ann_inp.place(relx=0.2, rely=0.4)

an_tit_1 = Label(window,text='散射角1(°)：')
an_tit_1.place(relx=0.05, rely=0.47)
an_inp_1 = Entry(window)
an_inp_1.place(relx=0.2, rely=0.47)

global Gspath_1
def selectpath_gs1():
    global Gspath_1
    Gspath_1 = askopenfilename()
    path_Gs_1.set(Gspath_1)

path_Gs_1 = StringVar()

path_Gs_1_tit = Label(window,text='对应光强自相关函数文件：')
path_Gs_1_tit.place(relx=0.05, rely=0.52)
path_Gs_1_inp = Entry(window, textvariable=path_Gs_1)
path_Gs_1_inp.place(relx=0.2, rely=0.57)
path_Gs_1_btn = Button(window, text='浏览文件', command=selectpath_gs1)
path_Gs_1_btn.place(relx=0.12, rely=0.56)

an_tit_2 = Label(window, text='散射角2(°)：')
an_tit_2.place(relx=0.05, rely=0.64)
an_inp_2 = Entry(window)
an_inp_2.place(relx=0.2, rely=0.64)

global Gspath_2
def selectpath_gs2():
    global Gspath_2
    Gspath_2 = askopenfilename()
    path_Gs_2.set(Gspath_2)

path_Gs_2 = StringVar()

path_Gs_2_tit = Label(window, text='对应光强自相关函数文件：')
path_Gs_2_tit.place(relx=0.05, rely=0.69)
path_Gs_2_inp = Entry(window, textvariable=path_Gs_2)
path_Gs_2_inp.place(relx=0.2, rely=0.74)
path_Gs_2_btn = Button(window, text='浏览文件', command=selectpath_gs2)
path_Gs_2_btn.place(relx=0.12, rely=0.73)

an_tit_3 = Label(window, text='散射角3(°)：')
an_tit_3.place(relx=0.05, rely=0.81)
an_inp_3 = Entry(window)
an_inp_3.place(relx=0.2, rely=0.81)

def selectpath_gs3():
    global Gspath_3
    Gspath_3 = askopenfilename()
    path_Gs_3.set(Gspath_3)

path_Gs_3 = StringVar()

path_Gs_3_tit = Label(window, text='对应光强自相关函数文件：')
path_Gs_3_tit.place(relx=0.05, rely=0.86)
path_Gs_3_inp = Entry(window, textvariable=path_Gs_3)
path_Gs_3_inp.place(relx=0.2, rely=0.91)
path_Gs_3_btn = Button(window, text='浏览文件', command=selectpath_gs3)
path_Gs_3_btn.place(relx=0.12, rely=0.9)

result = Label(text='反演结果', bg='white', width=15, height=3)
result.place(relx=0.6, rely=0.15)

def multiangle_mdls_exep():
    # 参数输入
    nm = float(nm_inp.get())                    #介质折射率
    np = float(np_inp.get())                     #颗粒折射率
    lm = float(lm_inp.get()) * 10 ** (-9)     #波长
    T = float(temp_inp.get()) + 273.15
    Dmin = float(Dmin_inp.get())*10**(-9)     #反演粒径下限
    Dmax = float(Dmax_inp.get())*10**(-9)   #反演粒径上限
    N = int(N_inp.get())

    D = linspace(Dmin, Dmax, N)  # linspace: 创建Dmin到Dmax的等差数列
    n1 = 0.874 * 10 ** (-3)     # n1为介质黏度系数，单位：g/nms(Pa·S)，即0.89cP
    Kb = 1.38 * 10 ** (-23)  # Kb为玻尔兹曼常数，单位：J/K
    phi = pi/2                      # phi为方位角，即偏振光的偏振角，取π/2)
    R = int(ann_inp.get())

    # 对实验所得acf(自相关)曲线进行拟合
    # 单峰曲线模拟
    def E_fitting1(acf1, tau):      # tau：延迟时间

        def f(x, A, B):
            return A * exp(B * x)

        acf1 = array(acf1)
        tau = array(tau)
        A1, B1 = curve_fit(f, tau, acf1)[0]
        fittingacf = A1 * exp(B1 * tau)
        plt.plot(tau, fittingacf, "blue")

        return fittingacf

    # 双峰模拟
    def E_fitting2(acf1, tau):      # 输入光强ACF

        def f(x, A1, B1, A2, B2):
            return A1 * exp(B1 * x) + A2 * exp(B2 * x)

        len1 = len(acf1)
        len2 = len(tau)

        t = []
        for i in range(0, len1):
            t.append(tau[i])

        acf = array(acf1)
        t = array(t)
        A1, B1, A2, B2 = curve_fit(f, t, acf, maxfev=50000)[0]
        tau = array(tau)
        fittingacf = A1 * exp(B1 * tau) + A2 * exp(B2 * tau)    # 拟合后的ACF曲线
        #plt.plot(tau, fittingacf, "blue")

        return fittingacf

    # 计算基线值
    # 使用测量基线值

    def gst(Gs, G0):  # 计算归一化电场自相关函数；G0 = EG_infinite_u，为基线值
        b = 1  # b为散射光场的相干度β，取1
        gs = []  # gs为归一化的电场自相关函数g(τ)
        for G in Gs:
            #g = sqrt((abs((G/G0) - 1))/b)
            g = sqrt((abs(G / b)))
            gs.append(g)
        return gs

    # 计算米散射光强
    def mie(a, lm, m, ang, phi):
        # a为颗粒粒度（直径），lm为波长
        # m为复折射率(散射颗粒相对于周围介质的折射率）
        # ang为散射角,phi为方位角
        k0 = pi / lm  # k0为真空波数，k0=2π/λ
        x = k0 * a  # x为尺寸参数，x=ko*a
        u = cos(ang)  # u为散射角的余弦值

        nmax = round(2 + x + 4 * x ** (1 / 3))

        # 计算an，bn
        z = m * x
        nmx = round(max(nmax, abs(z)) + 16)
        n = list(range(1, nmax + 1))
        nu = []
        xs = []
        for ns in n:
            nu.append(ns + 0.5)
            xs.append(x)

        sx = sqrt(0.5 * pi * x)
        px = []
        bse1 = special.jv(nu, xs)
        for bs in bse1:
            px.append(sx * bs)

        p1x = [sin(x)]
        for i in range(0, nmax - 1):
            p1x.append(px[i])

        chx = []
        bse2 = special.yv(nu, xs)
        for bs in bse2:
            chx.append(-sx * bs)

        ch1x = [cos(x)]
        for i in range(0, nmax - 1):
            ch1x.append(chx[i])

        gsx = list(range(0, nmax))
        for i in range(0, nmax):
            gsx[i] = px[i] - 1j * chx[i]

        gs1x = list(range(0, nmax))
        for i in range(0, nmax):
            gs1x[i] = p1x[i] - 1j * ch1x[i]

        dnx = list(range(0, nmx))
        dnx[nmx - 1] = 0 + 0 * 1j
        for i in range(nmx - 1, 0, -1):
            dnx[i - 1] = (i + 1) / z - 1 / (dnx[i] + (i + 1) / z)

        da = list(range(0, nmax))
        db = list(range(0, nmax))
        an = list(range(0, nmax))
        bn = list(range(0, nmax))
        for i in range(0, nmax):
            da[i] = dnx[i] / m + (i + 1) / x
            db[i] = m * dnx[i] + (i + 1) / x
            an[i] = (da[i] * px[i] - p1x[i]) / (da[i] * gsx[i] - gs1x[i])
            bn[i] = (db[i] * px[i] - p1x[i]) / (db[i] * gsx[i] - gs1x[i])

        pin = list(range(0, nmax))
        tin = list(range(0, nmax))
        pin[0] = 1
        tin[0] = u
        pin[1] = 3 * u
        tin[1] = 3 * cos(2 * a * cos(u))
        for i in range(2, nmax):
            p1 = (2 * (i + 1) - 1) / i * pin[i - 1] * u
            p2 = (i + 1) / i * pin[i - 1]
            pin[i] = p1 - p2

            t1 = (i + 1) * u * pin[i]
            t2 = (i + 2) * pin[i - 1]
            tin[i] = t1 - t2

        S1 = 0
        S2 = 0
        for i in range(1, nmax + 1):
            n2 = (2 * i + 1) / (i * (i + 1))
            pin[i - 1] = n2 * pin[i - 1]
            tin[i - 1] = n2 * tin[i - 1]
            S1 = S1 + an[i - 1] * pin[i - 1] + bn[i - 1] * tin[i - 1]
            S2 = S2 + an[i - 1] * tin[i - 1] + bn[i - 1] * pin[i - 1]

        # 计算粒度为D在散射角an处的散射光强
        I0 = lm ** 2 * ((abs(S1) ** 2) * (sin(phi) ** 2) + (abs(S2) ** 2) * (cos(phi)) ** 2)
        return I0

    # Tikhonov正则化方法求解矩阵方程，求粒径分布f
    def tikh(G, b):     # Af = g，tikh(A, g)
        # 求best_param(最佳正则化参数)
        npoints = 150
        smin_ratio = 16 * spacing(1)    # 返回x与最近相邻数的间距
        reg_param = linspace(10 ** (-8), 0.01, npoints)     #创建等差数列，正则化参数在这里面选
        g1 = G.shape[0]     # 求第一层数组的元素个数（第一层：即最外层）
        g2 = G.shape[1]     # 求第二层数组的元素个数
        eta = []
        for i in range(0, npoints):
            reg_param_c = reg_param[i]  # 即λ
            c1 = G
            c2 = reg_param_c * eye(g2)  # eye：生成 g2×g2 的单位矩阵
            C = bmat([[c1], [c2]])  # bmat：组成分块矩阵，c1、c2竖向组合
            d1 = b
            d1 = d1.reshape(len(d1), 1)  # 将数组转化为m行n列的矩阵
            d2 = zeros((g2, 1))              # 形成数组，g2行1列，元素全为0
            d = bmat([[d1], [d2]])
            d0 = []
            for j in range(0, len(d)):
                d0.append(d[(j, 0)])  # d0 = b？
            x, x_norm = nnls(C, d0)  # 非负最小二乘求解
            eta.append(linalg.norm(x))  # linalg.norm: 求范数
        q = eta     # q存储最小二程解的范数，即所有的||xλ||
        q2 = []      # q2为q元素的平方，即||xλ||2
        for qs in q:
            q2.append(qs ** 2)
        dq = diff(q2)       # diff：输出相邻两元素的差
        dq = insert(dq, npoints - 1, dq[npoints - 2])   # 在npoints - 1位置插入dq[npoints - 2]
        ddq = diff(dq)
        ddq = insert(ddq, npoints - 1, ddq[npoints - 2])
        k = []
        for i in range(0, npoints):
            k.append(abs(ddq[i]) / (1 + dq[i] ** 2) ** 1.5)     # 曲率k=|y''| / (1 + y' ** 2) ** 1.5

        # MR-L曲线法
        eta_2 = []
        for etas in eta:
            eta_2.append(etas ** 2)
        reg_param_2 = []
        for r in reg_param:
            reg_param_2.append((r ** 2))

        v = list(range(0, npoints))  # range：生成0,1,2……npoints-1
        for i in range(0, npoints - 1):
            v[i] = (k[i + 1] - k[i]) / (reg_param[i + 1] - reg_param[i])  # ∆k/∆λ，曲率的变化曲线
        v[npoints - 1] = v[npoints - 2]
        V = []
        for vs in v:
            V.append(abs(vs))       # V为v取绝对值
        v_number = argmax(V)    # 返回最大元素的位置
        best_param = reg_param[v_number]  # best_param为正则化参数
        eta_c = eta[v_number]
        k_c = k[v_number]
        v_c = v[v_number]

        c1 = G
        c2 = best_param * eye(g2)
        C = bmat([[c1], [c2]])
        d1 = b
        d1 = d1.reshape(len(d1), 1)
        d2 = zeros((g2, 1))
        d = bmat([[d1], [d2]])
        d0 = []
        for j in range(0, len(d)):
            d0.append(d[(j, 0)])
        Xr, Xr_norm = nnls(C, d0)

        return Xr   # 返回粒径分布

    # 求散射光强分数Cdi
    def cdi(an):
        #I = list(range(0, N))   # I = [0,1,2···N-1]
        I = []
        for i in range(0, N):
            a = D[i]    # a为粒度
            I.append(mie(a, lm, np / nm, an, phi))
        I_sum = sum(I)  # 求和函数
        Cdi = []  # Cdi = list(range(0, N))
        for i in range(0, N):
            Cdi.append(round(I[i] / I_sum, 6))  # 四舍五入
        return Cdi

    # 计算系数矩阵A, Af = g
    def Fr(T0, D, Cdi):
        r_1 = []
        for t in times_1:   # times_1:文件第一列
            r_1.append(-T0 * t)     # T0: Γ0
        R_1 = array(r_1)
        R_1 = R_1.reshape(len(R_1), 1)

        r_2 = []
        for Di in D:
            r_2.append(1 / Di)
        R_2 = array(r_2)
        R_2 = R_2.reshape(1, len(R_2))
        r_3 = dot(R_1, R_2)     # dot：向量乘法/矩阵乘法 -Γ0 * t / Di
        R_3 = exp(r_3)    # exp(-Γ0 * t / Di)
        mat_cdi = diag(Cdi)  # diag:一维数组：形成以一维数组为对角线元素的矩阵；二维：输出对角线元素
        A = dot(R_3, mat_cdi)   # exp(-Γ0 * t / Di) * Cdi
        return A

    # 多角度结合，迭代递归算法
    def multiangle(R, acf1, acf2, acf3, an, tau):   # an是散射角度（弧度制）
        if R == 2:
            # 第1个角度
            z1 = E_fitting2(acf1, tau)           # 实验数据拟合
            G0_1 = E_G_infinite_u(z1, tau)  # 计算基线值
            gs_1 = gst(z1, G0_1)                # 由基线值和G计算g
            # 第2个角度
            z2 = E_fitting2(acf2, tau)           # 实验数据拟合
            G0_2 = E_G_infinite_u(z2, tau)  # 计算基线值
            gs_2 = gst(z2, G0_2)                # 由基线值和G计算g

            gs_new = gs_1 + gs_2     # list相加
            T01 = TT(an[0], nm, lm)    # 计算Γ0
            Cdi1 = cdi(an[0])              # 求散射光强分数Cdi
            A = Fr(T01, D, Cdi1)        # 第一个角度的系数矩阵

        else:   # R若不是2，则只能是3
            # 第1个角度
            z1 = E_fitting2(acf1, tau)  # 实验数据拟合
            G0_1 = E_G_infinite_u(z1, tau)  # 计算基线值
            gs_1 = gst(z1, G0_1)  # 由基线值和G计算g
            # 第2个角度
            z2 = E_fitting2(acf2, tau)  # 实验数据拟合
            G0_2 = E_G_infinite_u(z2, tau)  # 计算基线值
            gs_2 = gst(z2, G0_2)  # 由基线值和G计算g
            # 第3个角度
            z3 = E_fitting2(acf3, tau)  # 实验数据拟合
            G0_3 = E_G_infinite_u(z3, tau)  # 计算基线值
            gs_3 = gst(z3, G0_3)  # 由基线值和G计算g

            gs_new = gs_1 + gs_2 + gs_3  # list相加
            T01 = TT(an[0], nm, lm)   # 计算Γ0
            Cdi1 = cdi(an[0])              # 求散射光强分数Cdi
            A = Fr(T01, D, Cdi1)        # 第一个角度的系数矩阵

        # 利用scipy中的非线性规划来求取最佳k（权重系数）
        def f(k):
            vec = gsi - k * dot(A_i, f_1)           # dot: 向量/矩阵乘法
            return dot(transpose(vec), vec)     # transpose: 调换行列的索引值，类似求转置



            k_i = optimize.minimize(f, 0, bounds=[(0, 1)])     # 非线性规划求极小值，f：需最小化的目标函数、x0：初始值、bounds：变量的边界, (min,max)
            print("第" + str(i) + "个角度权重系数" + str(k_i.x))
            Ai_k = k_i.x * A_i      # k_i.f: 返回方程f的最小值；k_i.x: 返回方程最小时的k值
            A_new = bmat([[A], [Ai_k]])     # 联合矩阵
            gs_i = gs_new[0:i * t_len]       # 前i个角度的电场自相关函数（联合自相关函数）
            gs_i = array(gs_i)
            f_i = tikh(A_new, gs_i)    # 返回粒径分布

            return f_i

    # 寻找峰值并判断是单峰分布还是双峰分布
    def find_peak(A, B):    # find_peak(f0, D)
        A1 = diff(A)
        A2 = []
        for a1 in A1:
            A2.append(sign(a1))
        A3 = diff(A2)
        A3 = A3.tolist()
        n = A3.count(-2)  # n为极大值的个数，n=1 判断为单峰，n=2判断为双峰
        if n == 0:
            max_num = argmax(A)
            A_peak = round((B[max_num] * 10 ** 9), 2)
            f_peak_tit = Label(window, text='反演最大粒径为：' + str(A_peak) + ' nm')
            f_peak_tit.place(relx=0.45, rely=0.25)
        if n == 2:
            n1 = A3.index(-2) + 1  # n1为第一个极大值所在位置
            max_num = argmax(A)
            n2 = A3.index(-2, n1 + 1)
            A_peak1 = round((B[n1] * 10 ** 9), 2)
            A_peak2 = round((B[n2] * 10 ** 9), 2)
            f_peak_tit = Label(window, text='反演粒径分别为：'+str(A_peak1)+' nm 与 '+str(A_peak2)+' nm')
            f_peak_tit.place(relx=0.45, rely=0.25)
            if max_num != n1 or n2:
                A_max = round((B[max_num] * 10 ** 9), 2)
                f_peak_tit2 = Label(window, text='反演最大粒径为：' + str(A_max) + ' nm ')
                f_peak_tit2.place(relx=0.45, rely=0.3)
        if n == 1:
            n1 = A3.index(-2) + 1  # n1为第一个极大值所在位置
            max_num = argmax(A)
            A_peak = round((B[n1] * 10 ** 9), 2)
            #A_peak_ = round((A_peak * 10 ** (9)), 2)
            f_peak_tit = Label(window, text='反演峰值粒径为：' + str(A_peak) + ' nm')
            f_peak_tit.place(relx=0.45, rely=0.25)
            if max_num != n1:
                A_max = round((B[max_num] * 10 ** 9), 2)
                f_peak_tit2 = Label(window, text='反演最大粒径为：' + str(A_max) + ' nm ')
                f_peak_tit2.place(relx=0.45, rely=0.3)

    def gscut(gs):
        gs_cut = []
        max = gs[0]
        for g in gs:
            if g > 0.2 * max:
                gs_cut.append(g)
        return gs_cut


    #单角度时
    if R == 1:
        an = float(an_inp_1.get()) * pi / 180   # 化为弧度
        T0 = 16 * pi * nm * nm * Kb * T * ((sin(an / 2)) ** 2) / (3 * n1 * lm * lm)  # 计算T0

        # 读取实验所得相关函数文件
        times = []
        Gs_ori = []
        with open(str(Gspath_1), 'r', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                i += 1
                if i > 4:
                    times.append(round(float(row[0]) * 10 ** (-6), 7))
                    #Gs_ori.append(float(row[2]))
                    Gs_ori.append(float(row[1]))

        #Gs = E_fitting2(Gs_ori, times)  # 返回光强自相关曲线
        B = (Gs_ori[-1]+Gs_ori[-2]+Gs_ori[-3]+Gs_ori[-4]) / 4

        # 对比拟合g2与否的结果
        gs_ori = gst(Gs_ori, B)                 # 计算原始电场ACF
        gs_cut = gscut(gs_ori)
        gs_fit = E_fitting2(gs_cut, times)   # 计算拟合电场ACF

        #plt.plot(times, Gs_ori, "red", linewidth=6)
        #plt.plot(times, gs_ori, "blue", linewidth=3)
        #plt.xscale('log')
        #plt.show()

        Cdi = cdi(an)
        # 最小二乘求解方程Af = g
        gt = array(gs_ori)
        #gt = array(gs_fit)

        # 计算系数矩阵A(Fr.m)
        R_1 = []
        for t in times:
            R_1.append(-T0 * t)
        r_1 = array(R_1)
        r_1 = r_1.reshape(len(r_1), 1)

        R_2 = []
        for Di in D:
            R_2.append(1 / Di)
        r_2 = array(R_2)
        r_2 = r_2.reshape(1, len(r_2))
        R_3 = dot(r_1, r_2)
        r_3 = exp(dot(r_1, r_2))
        r_4 = Cdi
        r_5 = diag(Cdi)
        A = dot(r_3, r_5)

        f0 = tikh(A, gt)    # 得出粒径分布

        # 找出f(Di)的最大值及其对应的Di值，即为峰值粒径
        Di_index = argmax(f0)   # 返回最大元素的位置

        # 归一化得出粒径分布
        f0_max = max(f0)    # 返回最大的元素
        f_final = []
        for f0s in f0:
            f_final.append(f0s / f0_max)    # 归一化

        find_peak(f0, D)


    # 双角度时
    elif R == 2:
        an = [float(an_inp_1.get()) * pi / 180, float(an_inp_2.get()) * pi / 180]

        # 读取实验所得相关函数文件
        # 第1个角度
        times_1 = []
        Gs_ori_1 = []
        print(str(Gspath_1))
        with open(str(Gspath_1)) as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                i += 1
                if i > 4:
                    times_1.append(round(float(row[0]) * 10 ** (-6), 7))
                    Gs_ori_1.append(float(row[1]))

        # 第2个角度
        times_2 = []
        Gs_ori_2 = []
        print(str(Gspath_2))
        with open(str(Gspath_2), 'r', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                i += 1
                if i > 4:
                    times_2.append(round(float(row[0]) * 10 ** (-6), 7))
                    Gs_ori_2.append(float(row[1]))

        Gs_ori_3 = []

        f0 = multiangle(R, Gs_ori_1, Gs_ori_2, Gs_ori_3, an, times_1)

        # 找出f(Di)的最大值及其对应的Di值，即为峰值粒径
        Di_index = argmax(f0)

        # 归一化得出粒径分布
        f0_max = max(f0)
        f_final = []
        for f0s in f0:
            f_final.append(f0s / f0_max)

        find_peak(f0, D)

    # 三角度时
    else:    # R == 3
        an = [float(an_inp_1.get()) * pi / 180, float(an_inp_2.get()) * pi / 180, float(an_inp_2.get()) * pi / 180]

        # 读取实验所得相关函数文件
        # 第1个角度
        times_1 = []
        Gs_ori_1 = []
        print(str(Gspath_1))
        with open(str(Gspath_1)) as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                i += 1
                if i > 4:
                    times_1.append(round(float(row[0]) * 10 ** (-6), 7))
                    Gs_ori_1.append(float(row[1]))

        # 第2个角度
        times_2 = []
        Gs_ori_2 = []
        print(str(Gspath_2))
        with open(str(Gspath_2)) as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                i += 1
                if i > 4:
                    times_2.append(round(float(row[0]) * 10 ** (-6), 7))
                    Gs_ori_2.append(float(row[1]))

        # 第3个角度
        times_3 = []
        Gs_ori_3 = []
        print(str(Gspath_3))
        with open(str(Gspath_3)) as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                i += 1
                if i > 4:
                    times_3.append(round(float(row[0]) * 10 ** (-6), 7))
                    Gs_ori_3.append(float(row[1]))

        f0 = multiangle(R, Gs_ori_1, Gs_ori_2, Gs_ori_3, an, times_1)

        # 找出f(Di)的最大值及其对应的Di值，即为峰值粒径
        Di_index = argmax(f0)

        # 归一化得出粒径分布
        f0_max = max(f0)
        f_final = []
        for f0s in f0:
            f_final.append(f0s / f0_max)

        find_peak(f0, D)

    # 画出粒径分布曲线
    plt.plot(D * 10 ** 9, f_final, 'blue', label="PSD")
    plt.legend(loc="right")  # 放置位置
    #plt.rcParams['font.sans-serif'] = ['SimHei']  # 为图标输出中文
    #plt.rcParams['axes.unicode_minus'] = False
    #plt.title('Python反演粒径分布')
    plt.xlim(D[0] * 10 ** 9, D[-1] * 10 ** 9)
    plt.xlabel('D(nm)')
    plt.ylabel('f(Di)')
    plt.savefig("inversion_curve.png")  # 反演曲线
    plt.clf()

    # 显示反演曲线图形和峰值粒径
    txt = Text(window, height=30, width=70)
    global photo   # ?
    pho = Image.open("inversion_curve.png")
    pho = pho.resize((480, 360))
    photo = ImageTk.PhotoImage(pho)
    txt.insert(END, '\n')
    txt.image_create(END, image=photo)
    txt.place(relx=0.4, rely=0.35)




