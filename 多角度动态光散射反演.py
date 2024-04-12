import matplotlib.pyplot as plt
from scipy import special
from scipy import optimize,linalg
from scipy.optimize import curve_fit
from numpy import exp,pi,sin,cos,argsort,zeros,mean,linspace,sum,array,matrix,bmat,argmax,transpose,dot,diag
from numpy import diff,insert,array,spacing,eye,identity,matmul,sign
from scipy.optimize import nnls
import numpy as np
import csv
from math import sqrt
#from sklearn.linear_model import Lasso
from tkinter import *
from tkinter.filedialog import *
from PIL import Image,ImageTk

#实验数据文件输入,单角度/双角度/三角度
#用于申请软件著作权

window = Tk()
window.title("多角度动态光散射粒径分析")
window.geometry('1200x700')

#参数输入界面
nm_tit = Label(window,text='分散介质折射率：')
nm_tit.place(relx=0.05,rely=0.05)
nm_inp = Entry(window)
nm_inp.place(relx=0.2,rely=0.05)

np_tit = Label(window,text='颗粒折射率：')
np_tit.place(relx=0.05,rely=0.1)
np_inp = Entry(window)
np_inp.place(relx=0.2,rely=0.1)

lm_tit = Label(window,text='入射光波长(nm)：')
lm_tit.place(relx=0.05,rely=0.15)
lm_inp = Entry(window)
lm_inp.place(relx=0.2,rely=0.15)

temp_tit = Label(window,text='温度(°C)：')
temp_tit.place(relx=0.05,rely=0.2)
temp_inp = Entry(window)
temp_inp.place(relx=0.2,rely=0.2)

Dmin_tit = Label(window,text='反演范围下限(nm)：')
Dmin_tit.place(relx=0.05,rely=0.25)
Dmin_inp = Entry(window)
Dmin_inp.place(relx=0.2,rely=0.25)

Dmax_tit = Label(window,text='反演范围上限(nm)：')
Dmax_tit.place(relx=0.05,rely=0.3)
Dmax_inp = Entry(window)
Dmax_inp.place(relx=0.2,rely=0.3)

N_tit = Label(window,text='取样点数：')
N_tit.place(relx=0.05,rely=0.35)
N_inp = Entry(window)
N_inp.place(relx=0.2,rely=0.35)

ann_tit = Label(window,text='散射角个数：')
ann_tit.place(relx=0.05,rely=0.4)
ann_inp = Entry(window)
ann_inp.place(relx=0.2,rely=0.4)

an_tit_1 = Label(window,text='散射角1(°)：')
an_tit_1.place(relx=0.05,rely=0.45)
an_inp_1 = Entry(window)
an_inp_1.place(relx=0.2,rely=0.45)

def selectpath_gs1():
    global Gspath_1
    Gspath_1 = askopenfilename()
    path_Gs_1.set(Gspath_1)

path_Gs_1 = StringVar()

path_Gs_1_tit = Label(window,text='散射角1对应光强自相关函数：')
path_Gs_1_tit.place(relx=0.05,rely=0.5)
path_Gs_1_inp = Entry(window,textvariable=path_Gs_1)
path_Gs_1_inp.place(relx=0.13,rely=0.55)
path_Gs_1_btn = Button(window,text='浏览文件',command=selectpath_gs1)
path_Gs_1_btn.place(relx=0.26,rely=0.55)

an_tit_2 = Label(window,text='散射角2(°)：')
an_tit_2.place(relx=0.05,rely=0.6)
an_inp_2 = Entry(window)
an_inp_2.place(relx=0.2,rely=0.6)

def selectpath_gs2():
    global Gspath_2
    Gspath_2 = askopenfilename()
    path_Gs_2.set(Gspath_2)

path_Gs_2 = StringVar()

path_Gs_2_tit = Label(window,text='散射角2对应光强自相关函数：')
path_Gs_2_tit.place(relx=0.05,rely=0.65)
path_Gs_2_inp = Entry(window,textvariable = path_Gs_2)
path_Gs_2_inp.place(relx=0.13,rely=0.7)
path_Gs_2_btn = Button(window,text='浏览文件',command=selectpath_gs2)
path_Gs_2_btn.place(relx=0.26,rely=0.7)

an_tit_3 = Label(window,text='散射角3(°)：')
an_tit_3.place(relx=0.05,rely=0.75)
an_inp_3 = Entry(window)
an_inp_3.place(relx=0.2,rely=0.75)

def selectpath_gs3():
    global Gspath_3
    Gspath_3 = askopenfilename()
    path_Gs_3.set(Gspath_3)

path_Gs_3 = StringVar()

path_Gs_3_tit = Label(window,text='散射角3对应光强自相关函数：')
path_Gs_3_tit.place(relx=0.05,rely=0.8)
path_Gs_3_inp = Entry(window,textvariable = path_Gs_3)
path_Gs_3_inp.place(relx=0.13,rely=0.85)
path_Gs_3_btn = Button(window,text='浏览文件',command=selectpath_gs3)
path_Gs_3_btn.place(relx=0.26,rely=0.85)

out_tit = Label(window,text='反演结果',bg='white',width=15, height=2).place(relx=0.6,rely=0.15)



def multiangle_mdls_exep():
    # 参数输入
    nm = float(nm_inp.get())
    np = float(np_inp.get())
    lm = float(lm_inp.get()) * 10 ** (-9)
    T = float(temp_inp.get()) + 273.15
    Dmin = float(Dmin_inp.get())*10**(-9)
    Dmax = float(Dmax_inp.get())*10**(-9)
    N = int(N_inp.get())

    n1 = 0.89 * 10 ** (-3)  # n1为介质黏度系数，单位：g/nms
    Kb = 1.38 * 10 ** (-23)  # Kb为玻尔兹曼常数，单位：J/K
    phi = pi/2 # phi为方位角（偏振光的偏振角，取π/2)
    R = int(ann_inp.get())

    def TT(an, nm, lm):  # 计算T0
        Kb = 1.38 * 10 ** (-23)
        n1 = 0.89 * 10 ** (-3)
        T0 = 16 * pi * nm * nm * Kb * T * ((sin(an / 2)) ** 2) / (3 * n1 * lm * lm)
        return T0

    # 对实验所得acf曲线进行拟合
    def E_fitting1(acf1, tau):  # 单峰曲线模拟
        acf = []
        e_gr = []
        acf = bmat([[acf], [acf1]])

        def f(x, A, B):
            return A * exp(B * x)

        acf1 = array(acf1)
        tau = array(tau)
        A1, B1 = optimize.curve_fit(f, tau, acf1)[0]
        fittingacf = A1 * exp(B1 * tau)
        plt.plot(tau, fittingacf, "blue")

        return fittingacf

    # 双峰模拟
    def E_fitting2(acf1, tau):
        acf = []
        e_gr = []
        acf = bmat([[acf], [acf1]])

        def f(x, A1, B1, A2, B2):
            return A1 * exp(B1 * x) + A2 * exp(B2 * x)

        acf1 = array(acf1)
        tau = array(tau)
        A1, B1, A2, B2 = optimize.curve_fit(f, tau, acf1)[0]
        fittingacf = A1 * exp(B1 * tau) + A2 * exp(B2 * tau)
        #print(A1, B1, A2, B2)

        return fittingacf

    # 计算基线值
    def E_G_infinite_u(gs, times):  # gs:拟合后的光强ACF
        f_number = 10  # 直线拟合数据数目L取10
        gs_n = len(gs)
        Max = max(gs)
        Min = min(gs)
        # Halfheight = (Max - Min) / 2
        Halfheight = (Max + Min) / 2

        R = []
        for g in gs:
            R.append(abs(g - Halfheight))
        index = argsort(R)
        HalfHeight = gs[index[0]]
        HalfHeight_i = index[0]

        def f_1(x, A, B):
            return A * x + B

        plt.figure
        x0 = times[HalfHeight_i:(HalfHeight_i + f_number)]
        y0 = gs[HalfHeight_i:(HalfHeight_i + f_number)]
        p1, p2 = optimize.curve_fit(f_1, x0, y0)[0]
        k0 = p1

        meany = zeros((gs_n - HalfHeight_i - f_number, 1))
        k = zeros((gs_n - HalfHeight_i - f_number, 1))

        for i in range(1, (gs_n - HalfHeight_i - f_number + 1)):
            x = []
            for xs in x0:
                x.append(xs + i)
            y = gs[(HalfHeight_i + i):(HalfHeight_i + f_number + i)]
            meany[i - 1, 0] = mean(y)
            p1, p2 = optimize.curve_fit(f_1, x, y, maxfev=20000, method='dogbox')[0]
            k[i - 1, 0] = p1

            if k[i - 1, 0] > ((0.0007) * k0):
                EG_infinite_u_x = times[HalfHeight_i + i - 1]
                EG_infinite_u = mean(y)

        if k[(gs_n - HalfHeight_i - f_number - 1), 0] <= ((0.0007) * k0):
            EG_infinite_u_x = times[gs_n - 1]
            y = gs[gs_n - 1]
            EG_infinite_u = y
        return EG_infinite_u

    def gst(Gs, G0):  # 计算归一化电场自相关函数
        b = 1  # b为散射光场的相干度β，取β=1
        gs = []  # gs为归一化的电场自相关函数g(τ)
        for G in Gs:
            g = sqrt((abs(G / G0 - 1))/b)
            gs.append(g)
        return gs

    # 计算mie散射光强
    def mie(a, lm, m, ang, phi):
        # a为颗粒粒度（直径），lm为波长
        # m为复杂折射率比(散射颗粒相对于周围介质的折射率）
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

    # Tikhonov方法 求解矩阵方程
    # 求best_param,最佳正则化参数
    def tikh(G, b):
        npoints = 150
        smin_ratio = 16 * spacing(1)
        reg_param = linspace(10 ** (-8), 0.01, npoints)
        g1 = G.shape[0]
        g2 = G.shape[1]
        eta = []
        for i in range(0, npoints):
            reg_param_c = reg_param[i]
            c1 = G
            c2 = reg_param_c * eye(g2)
            C = bmat([[c1], [c2]])
            d1 = b
            d1 = d1.reshape(len(d1), 1)
            d2 = zeros((g2, 1))
            d = bmat([[d1], [d2]])
            d0 = []
            for j in range(0, len(d)):
                d0.append(d[(j, 0)])
            x, x_norm = nnls(C, d0)  # 非负最小二乘求解
            eta.append(linalg.norm(x))
        q = eta
        q2 = []
        for qs in q:
            q2.append(qs ** 2)
        dq = diff(q2)
        dq = insert(dq, npoints - 1, dq[npoints - 2])
        ddq = diff(dq)
        ddq = insert(ddq, npoints - 1, ddq[npoints - 2])
        k = []
        for i in range(0, npoints):
            k.append(abs(ddq[i]) / (1 + dq[i] ** 2) ** 1.5)

        # 新L曲线法
        eta_2 = []
        for etas in eta:
            eta_2.append(etas ** 2)
        reg_param_2 = []
        for r in reg_param:
            reg_param_2.append((r ** 2))

        v = list(range(0, npoints))
        for i in range(0, npoints - 1):
            v[i] = (k[i + 1] - k[i]) / (reg_param[i + 1] - reg_param[i])
        v[npoints - 1] = v[npoints - 2]
        V = []
        for vs in v:
            V.append(abs(vs))
        v_number = argmax(V)
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

        return Xr

    # 求散射光强分数Cdi
    def cdi(an):
        I = list(range(0, N))
        # (C.m)
        for i in range(0, N):
            a = D[i]
            I[i] = mie(a, lm, np / nm, an, phi)
        I_sum = sum(I)
        Cdi = list(range(0, N))
        for i in range(0, N):
            Cdi[i] = round(I[i] / I_sum, 6)
        return Cdi

    # 计算系数矩阵A
    def Fr(T0, D, Cdi):
        R_1 = []
        for t in times_1:
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
        return A

    # 多角度结合，迭代递归算法
    def multiangle(R, acf1, acf2, acf3, an, tau):
        if R == 2:
            # 第1个角度
            z1 = E_fitting2(acf1, tau)  # 实验数据拟合
            G0_1 = E_G_infinite_u(z1, tau)  # 计算基线值
            gs_1 = gst(z1, G0_1)  # 由基线值和G计算g
            # 第2个角度
            z2 = E_fitting2(acf2, tau)  # 实验数据拟合
            G0_2 = E_G_infinite_u(z2, tau)  # 计算基线值
            gs_2 = gst(z2, G0_2)  # 由基线值和G计算g

            gs = gs_1 + gs_2
            T01 = TT(an[0], nm, lm)
            Cdi1 = cdi(an[0])
            A = Fr(T01, D, Cdi1)  # 第一个角度的系数矩阵

        if R == 3:
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

            gs = gs_1 + gs_2 + gs_3
            T01 = TT(an[0], nm, lm)
            Cdi1 = cdi(an[0])
            A = Fr(T01, D, Cdi1)  # 第一个角度的系数矩阵


        for i in range(2, R + 1):
            t_len = len(tau)
            gsi = gs[(i - 1) * t_len:i * t_len]  # bi,第i个角度的电场自相关函数
            gsi = array(gsi)
            gs_i_1 = gs[0:(i - 1) * t_len]  # 前i-1个角度的电场自相关函数
            gt_i_1 = array(gs_i_1)
            f_simu_1 = tikh(A, gt_i_1)
            ani = an[i - 1]
            T0i = TT(ani, nm, lm)
            Cdi = cdi(ani)
            Ai = Fr(T0i, D, Cdi)

            # 利用scipy中的非线性规划来求取最佳k_star
            def f(k):
                return dot(transpose(gsi - (k * dot(Ai, f_simu_1))), (gsi - (k * dot(Ai, f_simu_1))))  # fer
            k_star_i = optimize.minimize(f, 0, bounds=[(0, 1)])
            print("第" + str(i) + "个角度权重系数" + str(k_star_i.x))
            Ai_k = k_star_i.x * Ai
            A = bmat([[A], [Ai_k]])
            gs_i = gs[0:i * t_len]  # 前i个角度的电场自相关函数
            gs_i = array(gs_i)
            f_i = tikh(A, gs_i)
        return f_i

    #单角度时
    if R == 1:
        an = float(an_inp_1.get()) * pi / 180
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
                    Gs_ori.append(float(row[1]))

        Gs = E_fitting2(Gs_ori, times)  #Gs为E_gR，也为b，曲线拟合

        G0 = E_G_infinite_u(Gs, times)  #计算基线值

        gs = gst(Gs, G0)  # 计算归一化电场自相关函数

        D = []
        D = linspace(Dmin, Dmax, N)
        I = list(range(0, N))  # I 用来存放每一种颗粒度下的散射光强
        for i in range(0, N):
            a = D[i]
            I[i] = mie(a, lm, np / nm, an, phi)
        I_sum = sum(I)  # 在散射角an处,所有粒度颗粒的散射光强总和
        Cdi = list(range(0, N))  # Cdi为散射角an处，粒度为di的颗粒的散射光强分数
        for i in range(0, N):
            Cdi[i] = round(I[i] / I_sum, 6)

        # 最小二乘求解方程g=Af
        gt = array(gs)

        # 计算系数矩阵A（Fr.m)
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

        f0 = tikh(A, gt)

        # 找出f(Di)的最大值及其对应的Di值，即为峰值粒径
        Di_index = argmax(f0)
        D_peak = D[Di_index]
        # print("峰值粒径为："+str(D_peak))

        # 归一化得出粒径分布
        f0_max = max(f0)
        f_final = []
        for f0s in f0:
            f_final.append(f0s / f0_max)


        # 寻找峰值并判断是单峰分布还是双峰分布
        def find_peak(A, B):
            A1 = diff(A)
            A2 = []
            for a1 in A1:
                A2.append(sign(a1))
            A3 = diff(A2)
            A3 = A3.tolist()
            n1 = A3.index(-2) + 1  # n1为第一个极大值所在位置
            n = A3.count(-2)  # n为极大值的个数，n=1 判断为单峰，n=2判断为双峰
            if n == 2:
                n2 = A3.index(-2, n1 + 1)
                A_peak1 = B[n1]
                A_peak2 = B[n2]
                f_peak_tit = Label(window, text='反演为双峰分布，峰值粒径分别为：' + str(A_peak1) + ',' + str(A_peak2))
                f_peak_tit.place(relx=0.4, rely=0.25)
            else:
                n3 = argmax(A)
                A_peak = B[n3]
                # f_peak_tit = Label(window, text='反演峰值粒径为：' + str(A_peak))
                A_peak_ = round((A_peak * 10 ** (9)), 2)
                f_peak_tit = Label(window, text='反演峰值粒径为：' + str(A_peak_) + ' nm')
                f_peak_tit.place(relx=0.4, rely=0.25)



        find_peak(f0, D)

        # 画出粒径分布曲线
        plt.plot(D * 10 ** 9, f_final, 'blue', label="反演粒径分布")
        plt.legend(loc="right")  # 放置位置
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 为图标输出中文
        plt.rcParams['axes.unicode_minus'] = False
        plt.title('python反演粒径分布')
        plt.xlabel('D\ nm')
        plt.ylabel('f(Di)')
        plt.xlim(D[0] * 10 ** 9, D[-1] * 10 ** 9)
        #plt.ylim(0, 1)
        plt.savefig("fanyanquxian.png")
        # plt.show()
        plt.clf()

        # 显示反演曲线图形和峰值粒径
        txt = Text(window, height=30, width=70)
        global photo1
        pho = Image.open("fanyanquxian.png")
        pho = pho.resize((480, 360))
        photo1 = ImageTk.PhotoImage(pho)
        txt.insert(END, '\n')
        txt.image_create(END, image=photo1)
        txt.place(relx=0.4, rely=0.35)

        Label(window,text="粒径分布曲线：").place(relx=0.4,rely=0.3)



    if R == 2:
        an = []
        an.append(float(an_inp_1.get()) * pi / 180)
        an.append(float(an_inp_2.get()) * pi / 180)

        # 读取实验所得相关函数文件
        # 第1个角度
        times_1 = []
        Gs_ori_1 = []
        print(str(Gspath_1))
        with open(str(Gspath_1), 'r', encoding="utf-8") as csvfile:
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

        D = []
        D = linspace(Dmin, Dmax, N)

        f0 = multiangle(R, Gs_ori_1, Gs_ori_2, Gs_ori_3, an, times_1)

        # 找出f(Di)的最大值及其对应的Di值，即为峰值粒径
        Di_index = argmax(f0)
        D_peak = D[Di_index]

        # 归一化得出粒径分布
        f0_max = max(f0)
        f_final = []
        for f0s in f0:
            f_final.append(f0s / f0_max)

        # 寻找峰值并判断是单峰分布还是双峰分布
        def find_peak(A, B):
            A1 = diff(A)
            A2 = []
            for a1 in A1:
                A2.append(sign(a1))
            A3 = diff(A2)
            A3 = A3.tolist()
            n1 = A3.index(-2) + 1  # n1为第一个极大值所在位置
            n = A3.count(-2)  # n为极大值的个数，n=1 判断为单峰，n=2判断为双峰
            if n == 2:
                n2 = A3.index(-2, n1 + 1)
                A_peak1 = B[n1]
                A_peak2 = B[n2]
                f_peak_tit = Label(window, text='反演为双峰分布，峰值粒径分别为：' + str(A_peak1) + ',' + str(A_peak2))
                f_peak_tit.place(relx=0.4, rely=0.25)
            else:
                n3 = argmax(A)
                A_peak = B[n3]
                #f_peak_tit = Label(window, text='反演峰值粒径为：' + str(A_peak))
                A_peak_ = round((A_peak * 10 ** (9)), 2)
                f_peak_tit = Label(window, text='反演峰值粒径为：' + str(A_peak_) + ' nm')
                f_peak_tit.place(relx=0.4, rely=0.25)


        find_peak(f0, D)


        # 画出粒径分布曲线
        plt.plot(D * 10 ** (9), f_final, 'blue', label="反演粒径分布")
        plt.legend(loc="right")  # 放置位置
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 为图标输出中文
        plt.rcParams['axes.unicode_minus'] = False
        plt.title('python反演粒径分布')
        plt.xlabel('D\ nm')
        plt.ylabel('f(Di)')
        plt.savefig("fanyanquxian.png")
        # plt.show()
        plt.clf()

        # 显示反演曲线图形和峰值粒径
        txt = Text(window, height=30, width=70)
        global photo2
        pho = Image.open("fanyanquxian.png")
        pho = pho.resize((480, 360))
        photo2 = ImageTk.PhotoImage(pho)
        txt.insert(END, '\n')
        txt.image_create(END, image=photo2)
        txt.place(relx=0.4, rely=0.35)

        Label(window, text="粒径分布曲线：").place(relx=0.4, rely=0.3)


    if R == 3:
        an = []
        an.append(float(an_inp_1.get()) * pi / 180)
        an.append(float(an_inp_2.get()) * pi / 180)
        an.append(float(an_inp_2.get()) * pi / 180)

        # 读取实验所得相关函数文件
        # 第1个角度
        times_1 = []
        Gs_ori_1 = []
        print(str(Gspath_1))
        with open(str(Gspath_1), 'r', encoding="utf-8") as csvfile:
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


        #第3个角度
        times_3 = []
        Gs_ori_3 = []
        print(str(Gspath_3))
        with open(str(Gspath_3), 'r', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                i += 1
                if i > 4:
                    times_3.append(round(float(row[0]) * 10 ** (-6), 7))
                    Gs_ori_3.append(float(row[1]))

        D = []
        D = linspace(Dmin, Dmax, N)

        f0 = multiangle(R, Gs_ori_1, Gs_ori_2, Gs_ori_3, an, times_1)

        # 找出f(Di)的最大值及其对应的Di值，即为峰值粒径
        Di_index = argmax(f0)
        D_peak = D[Di_index]
        # print("峰值粒径为："+str(D_peak))

        # 归一化得出粒径分布
        f0_max = max(f0)
        f_final = []
        for f0s in f0:
            f_final.append(f0s / f0_max)

            # 寻找峰值并判断是单峰分布还是双峰分布
            def find_peak(A, B):
                A1 = diff(A)
                A2 = []
                for a1 in A1:
                    A2.append(sign(a1))
                A3 = diff(A2)
                A3 = A3.tolist()
                n1 = A3.index(-2) + 1  # n1为第一个极大值所在位置
                n = A3.count(-2)  # n为极大值的个数，n=1 判断为单峰，n=2判断为双峰
                if n == 2:
                    n2 = A3.index(-2, n1 + 1)
                    A_peak1 = B[n1]
                    A_peak2 = B[n2]
                    #f_peak_tit = Label(window, text='反演为双峰分布，峰值粒径分别为：' + str(A_peak1) + ',' + str(A_peak2))
                    f_peak_tit = Label(window, text='反演峰值粒径分别为：' + str(A_peak1))
                    f_peak_tit.place(relx=0.45, rely=0.25)
                else:
                    n3 = argmax(A)
                    A_peak = B[n3]
                    # f_peak_tit = Label(window, text='反演峰值粒径为：' + str(A_peak))
                    A_peak_ = round((A_peak * 10 ** (9)), 2)
                    f_peak_tit = Label(window, text='反演峰值粒径为：' + str(A_peak_) + ' nm')
                    f_peak_tit.place(relx=0.45, rely=0.25)


        find_peak(f0, D)


        # 画出粒径分布曲线
        plt.plot(D[100:] * 10 ** (9), f_final[100:], 'blue', label="反演粒径分布")
        plt.legend(loc="right")  # 放置位置
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 为图标输出中文
        plt.rcParams['axes.unicode_minus'] = False
        plt.title('python反演粒径分布')
        plt.xlabel('D\ nm')
        plt.ylabel('f(Di)')
        plt.savefig("fanyanquxian.png")
        #plt.show()
        plt.clf()

        # 显示反演曲线图形和峰值粒径
        txt = Text(window, height=30, width=70)
        global photo3
        pho = Image.open("fanyanquxian.png")
        pho = pho.resize((480, 360))
        photo3 = ImageTk.PhotoImage(pho)
        txt.insert(END, '\n')
        txt.image_create(END, image=photo3)
        txt.place(relx=0.45, rely=0.35)

        Label(window, text="粒径分布曲线：").place(relx=0.45, rely=0.3)

btn = Button(window,text='确定',command=multiangle_mdls_exep)
btn.place(relx=0.15,rely=0.9)

window.mainloop()
