clear all;

%M = csvread('D:\File\实验\粒径反演算法\取样点个数探索\362nm_acf.txt', 4, 0); % 第5行开始
M = csvread('D:\File\实验\可变光程\230428\456nmPS20mgmL90°1.5mm\1.fit', 4, 0); % 第5行开始

t_real = M(:,1);
g_real = M(:,2);
%g_real = M(:,2) - 1;

R = 3;
Theta = [pi/6, pi/4, pi/2];    %散射角个数;散射角度取值;
np = 1.59;                    %颗粒折射率
nm = 1.334;                  %介质折射率
m = np/nm;                  %复折射率 m=m'+im"
lamda = 532 *10^-9;    %波长532nm
phi = pi/4;                   %方位角、偏振光的偏振角
Kb = 1.38*10^-23;
T = 22.3 + 273.15;        %温度25
cp = 0.948 * 10^-3;     %黏度系数kg/m/s Pa*s【量纲差6个数量级，原为-9，已修改】
%greal=xlsread('800nmCuO3角度ACF-0411','A1:A200');

%指数
DLS1=[];
DLS2=[];
DLS3=[];
for j=1:1
    tau=[];
    g2=[];
    T0=16*pi*(nm^2)*Kb*T*((sin(Theta(R)/2))^2)/(3*cp*lamda^2);
    cut1=1;
    cut2=196;
    %截断操作
    %for k=1:196
    %    cut1=cut1+1;
    %   if g_real(k,j)<(g_real(1,j)*1.1)
    %      break
    %   end
    %end
    %for k=1:196
    %    cut2=cut2+1;
    %    if g_real(k,j)<(g_real(1,j)*0.0)
    %        break
    %    end
    %end

    g2=g_real(cut1:cut2,j);
    tau=t_real(cut1:cut2).*1e-6;
    
    %拟合初始值
    b1=[1, 100]; 
    b2=[1, 100, -1]; 
    b3=[1, 100, -1, -0.1]; 
    
    %拟合目标函数：累积量法前1/2/3项
    fun1 = @(K1,x)K1(1).*exp(-2.*K1(2).*x);
    fun2 = @(K2,x)K2(1).*exp(-2.*K2(2).*x + K2(3).*(x.^2)); 
    fun3 = @(K3,x)K3(1).*exp(-2.*K3(2).*x + K3(3).*(x.^2) + K3(4).*(x.^3));
    
    [Gamma1] = nlinfit(tau, g2, fun1, b1);
    [Gamma2] = nlinfit(tau, g2, fun2, b2);
    [Gamma3] = nlinfit(tau, g2, fun3, b3);
    
    xishu1 = Gamma1(2);
    xishu2 = Gamma2(2);
    xishu3 = Gamma3(2);
    
    DLS1(j) = T0 / xishu1;
    DLS2(j) = T0 / xishu2;
    DLS3(j) = T0 / xishu3;
    
    fprintf('一阶累积量法：%f nm\n', DLS1(j).*1e9);
    fprintf('二阶累积量法：%f nm\n', DLS2(j).*1e9);
    fprintf('三阶累积量法：%f nm\n', DLS3(j).*1e9);

end
