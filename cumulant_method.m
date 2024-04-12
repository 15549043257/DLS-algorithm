clear all;

%M = csvread('D:\File\ʵ��\���������㷨\ȡ�������̽��\362nm_acf.txt', 4, 0); % ��5�п�ʼ
M = csvread('D:\File\ʵ��\�ɱ���\230428\456nmPS20mgmL90��1.5mm\1.fit', 4, 0); % ��5�п�ʼ

t_real = M(:,1);
g_real = M(:,2);
%g_real = M(:,2) - 1;

R = 3;
Theta = [pi/6, pi/4, pi/2];    %ɢ��Ǹ���;ɢ��Ƕ�ȡֵ;
np = 1.59;                    %����������
nm = 1.334;                  %����������
m = np/nm;                  %�������� m=m'+im"
lamda = 532 *10^-9;    %����532nm
phi = pi/4;                   %��λ�ǡ�ƫ����ƫ���
Kb = 1.38*10^-23;
T = 22.3 + 273.15;        %�¶�25
cp = 0.948 * 10^-3;     %��ϵ��kg/m/s Pa*s�����ٲ�6����������ԭΪ-9�����޸ġ�
%greal=xlsread('800nmCuO3�Ƕ�ACF-0411','A1:A200');

%ָ��
DLS1=[];
DLS2=[];
DLS3=[];
for j=1:1
    tau=[];
    g2=[];
    T0=16*pi*(nm^2)*Kb*T*((sin(Theta(R)/2))^2)/(3*cp*lamda^2);
    cut1=1;
    cut2=196;
    %�ضϲ���
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
    
    %��ϳ�ʼֵ
    b1=[1, 100]; 
    b2=[1, 100, -1]; 
    b3=[1, 100, -1, -0.1]; 
    
    %���Ŀ�꺯�����ۻ�����ǰ1/2/3��
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
    
    fprintf('һ���ۻ�������%f nm\n', DLS1(j).*1e9);
    fprintf('�����ۻ�������%f nm\n', DLS2(j).*1e9);
    fprintf('�����ۻ�������%f nm\n', DLS3(j).*1e9);

end
