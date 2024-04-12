clc;clear all;close all;
a=61;Dg=linspace(400*10^-9,700*10^-9,a);	%����ֲ�����λ������nm*10^-9
b=11;delta=linspace(0.05,0.15,b);           %����ֲ���׼��
smin=1*10^-08;smax=10*10^-08;               %ƽ������Ѱ�ŷ�Χ[smin,smax]
c=10;spread=linspace(smin,smax,c);          %ƽ������Ѱ��
d=a*b;test=linspace(1,a*b,d);               %���γ�ȡһ����Ϊ��������
dmin=200*10^-9;dmax=1300*10^-9;             %���巴�����ȷ�Χ[d1,d2]nm*10^-9
n=221;D=linspace(dmin,dmax,n);              %������n������D�д洢n�������������ֵ�����ȷ�Χ[d1,d2]
R=12;Theta=linspace(3/18*pi,14/18*pi,R);    %ɢ��Ǹ���;ɢ��Ƕ�ȡֵ;
np=1.59;nm=1.3316;m=np/nm;                  %����������particle refractive index;��ɢ����������dispersion medium refractive index;complex refractive index m=m'+im"
lamda=532*10^-9;phi=pi/4;                   %����nm*10^-9;��λ�ǡ�ƫ����ƫ���
Kb=1.38*10^-23;T=298.15;eta=0.89*10^-3;     %������������J/K;�����¶�K;������ϵ��kg/m/s Pa*s�����ٲ�6����������ԭΪ-9�����޸ġ�
Tpath=5000;t=linspace(20*10^-5,1.0,Tpath);	%��С/����ӳ�ʱ��tj��ͨ����Tpath{��ͼ��40000},ģ��������һ��
time=t';xtime=[time,time.^2,time.^3,time.^4,time.^5,time.^6,time.^7,time.^8,time.^9,time.^10,time.^11,time.^12,time.^13,time.^14,time.^15,time.^16];%��Ͻ�ȡ�ݼ�����ߴ����������8�Σ�����4-16�Ρ�
noise=0.001;trainmax=01;                    %��ǿ����غ����е�����ˮƽ����Ӳ�ͬ��������ѵ�������������

tic
for i=1:length(Theta)
    for k=1:n
        x=nm*pi*D(k)/lamda;                     %size parameter x=2pi*a/lamda a=sphere radius
        u=cos(Theta(i));                        %u=cos(scattering angle)-1<=u<=1
        nmax=round(2+x+4*x^(1/3));              %�ضϴ���
        clear pai tao
        pai(1)=1;pai(2)=3*u;                    %pai(0)=0;pai(1)=1
        tao(1)=u;tao(2)=3*cos(2*acos(u));       %����6u*u-3δ�������
        for n1=3:nmax                           %n1 integer from 1 to nmax
            pai(n1)=(2*n1-1)./(n1-1).*pai(n1-1).*u-n1./(n1-1).*pai(n1-2);
            tao(n1)=n1*u.*pai(n1)-(n1+1).*pai(n1-1);%[�õ�pai tao]
        end
        nmx=round(max(nmax,abs(m.*x))+16);      %�ȶ���Ҫ��
        n0=(1:nmax);nu=(n0+0.5);
        px=sqrt(pi*x/2)*besselj(nu,x);          %����׵�һ�౴��������
        p1x=[sin(x),px(1:nmax-1)];              %��һ������
        chx=-sqrt(pi*x/2)*bessely(nu,x);        %����׵ڶ��౴��������
        ch1x=[cos(x),chx(1:nmax-1)];
        gsx=complex(px,chx);gs1x=complex(p1x,ch1x);%�����޸�Ϊ+��
        dnx(nmx)=complex(0,0);                  %ȡ0��Ϊ��ֵ
        for j=nmx:-1:2
            dnx(j-1)=j./m./x-1/(dnx(j)+j./m./x);%����������ֵ�ȶ�
        end
        dn=dnx(n0);                             % Dn(z),n=1~nmax
        an=((dn./m+n0./x).*px-p1x)./((dn./m+n0./x).*gsx-gs1x);
        bn=((m.*dn+n0./x).*px-p1x)./((m.*dn+n0./x).*gsx-gs1x);%�õ�[an bn]
        pin=(2*n0+1)./(n0.*(n0+1)).*pai;tin=(2*n0+1)./(n0.*(n0+1)).*tao;
        S1=(an*pin'+bn*tin');S2=(an*tin'+bn*pin');%�õ�[S1 S2]
        I(k)=lamda^2*((abs(S1).^2).*(sin(phi).^2)+(abs(S2).^2).*(cos(phi).^2));%[ɢ���theta��,D(i)���ȵ�ɢ���ǿ]
    end
    Fraction(i,:)=I./sum(I);                %ͬɢ���theta��,D(i)���ȵ�ɢ���ǿ/�������ȵ�ɢ���ǿ�ܺ�=D(i)���ȵ�ɢ���ǿ���� ά��1*n������
end

for cyclea=1:a
    for cycleb=1:b
        y=((10^(-6))./((delta(cycleb)*(sqrt(2*pi))).*D)).*(exp(-(log(D./Dg(cyclea))).^2./(2*(delta(cycleb)^2))));
        fm((cyclea-1)*b+cycleb)=max(y);             %��¼��һ���ֵ
        f(:,(cyclea-1)*b+cycleb)=y.';               %����һ�� [n������ֵ��Ӧ�ķֲ�����ֵ] ά��n*1��������
        for i=1:R
            G(i)=10^-6*((Fraction(i,:)*f(:,(cyclea-1)*b+cycleb))^2);%��ģ��������ȷֲ�f(Di)��ĳһ�Ƕ�theta����ʵ����ֵG_infinite_u
        end
        GR=[];
        for i=1:R 
            T0=16*pi*(nm^2)*Kb*T*((sin(Theta(i)/2))^2)/(3*eta*lamda^2);
            g=1.*exp((-T0.*t)'*(1./D))*diag(Fraction(i,:));     %����*ά��Tpath��1��������*1��n��������*�Խ��� ά��Tpath��n
            GR=cat(1,GR,g);   %�������� ϵ������GR ά��[(M1+M2+.....+MR)��n],MrΪͨ������Tpath�д�ţ�nΪ���ȷ�Χ�ڵ�ȡ����
        end
        gR=GR*f(:,(cyclea-1)*b+cycleb);             %[��һ���糡����غ���g]
        for i=1:R
            gg(:,i)=gR(Tpath*(i-1)+1:Tpath*i,1);
            T0=16*pi*(nm^2)*Kb*T*((sin(Theta(i)/2))^2)/(3*eta*lamda^2);%ͬ��������T0
            for train=1:trainmax
                epsilon_train=normrnd(0,1.0,Tpath,1);%������ֵΪ0����׼��1������1����̬�ֲ�����������ѵ����
                Gacf_train(:,i)=G(i)*(1+noise*epsilon_train+abs(gR(Tpath*(i-1)+1:Tpath*i,1)).^2);%beita=1
                %                 g_train(:,i)=gg(:,i);              %����������Ϊѵ����
                g_train(:,i)=abs((Gacf_train(:,i)./G(i))-1).^(1/2);
                coefficient_train(:,i)=regress(log(g_train(:,i)),xtime);
                DDLS_train(i,(train-1)*a*b+(cyclea-1)*b+cycleb)=-T0/coefficient_train(1,i);
            end
            epsilon=normrnd(0,1.0,Tpath,1);         %������ֵΪ0����׼��1������1����̬�ֲ����������ڲ��Լ�
            Gacf_test(:,i)=G(i)*(1+noise*epsilon+abs(gR(Tpath*(i-1)+1:Tpath*i,1)).^2);%beita=1
            %             g_test(:,i)=gg(:,i);                   %����������Ϊ���Լ�
            g_test(:,i)=abs((Gacf_test(:,i)./G(i))-1).^(1/2);
            coefficient_test(:,i)=regress(log(g_test(:,i)),xtime);
            DDLS_test(i,(cyclea-1)*b+cycleb)=-T0/coefficient_test(1,i);
            %DLS(i,(cyclea-1)*b+cycleb)=Fraction(i,:)*f(:,(cyclea-1)*b+cycleb)/(Fraction(i,:)*(f(:,(cyclea-1)*b+cycleb)./D'));
        end
    end
end
toc

f_train=[];
for train=1:trainmax
    f_train=cat(2,f_train,f);
end

tic
for cyclec=1:c
    for cycled=1:d
        Pstorage=DDLS_train;P=DDLS_test(:,test(cycled));%Pstorage(:,test(cycled))=[];       %ȡ�����޳�������������
        Qstorage=f_train;Q=Qstorage(:,test(cycled));%Qstorage(:,test(cycled))=[];           %ȡ�����޳���У�������
        net=newgrnn(Pstorage,Qstorage,spread(cyclec));PSD=net(P);
        LOOEE(cyclec,cycled)=sum((Q-PSD).^2)/(sum(Q.^2));                                   %J��ƽ�����/ԭ����ƽ����
        %LOOEE(cyclec,cycled)=sum((Q-PSD).^2)/(fm(cycled)^2);                               %2��ƽ�����/ft���ֵ^2
    end
end
% Pstorage=DDLS_train;Qstorage=f_train;
% for cyclec=1:c
%     net=newgrnn(Pstorage,Qstorage,spread(cyclec));
%     for cycled=1:d
%         P=DDLS_test(:,test(cycled));Q=Qstorage(:,test(cycled));
%         PSD=net(P);
%         LOOEE(cyclec,cycled)=sum((Q-PSD).^2)/(sum(Q.^2));                                 %J��ƽ�����/ԭ����ƽ����
%         %LOOEE(cyclec,cycled)=sum((Q-PSD).^2)/(fm(cycled)^2);                             %2��ƽ�����/ft���ֵ^2
%     end
% end
LOOCV=sum(LOOEE,2);                                                                         %ĳһspreadֵƽ�����֮�ͣ���¼����������ָ����ֵ
toc

[min1,position1]=min(LOOCV);                                                                %Ѱ������ƽ�����ӵ�λ��
[max2,position2]=max(LOOEE(position1,:));                                                   %Ѱ������ƽ�����Ӳ��Թ�����������ߵ�λ��
%position2=61;
Pstorage=DDLS_train;P=DDLS_test(:,test(position2));%Pstorage(:,test(position2))=[];         %ȡ�����޳�������������
Qstorage=f_train;Q=Qstorage(:,test(position2));%Qstorage(:,test(position2))=[];             %ȡ�����޳���У�������
net=newgrnn(Pstorage,Qstorage,spread(position1));PSD=net(P);                                %��ô�ʱ�ķ��ݽ��
[top1,position3]=max(PSD);PDg=dmin+(position3-1)*5*10^-9;                                   %Ѱ�ҷ���PSD��ֵλ��
[top2,position4]=max(Q);QDg=dmin+(position4-1)*5*10^-9;                                     %Ѱ������Q��ֵλ��
re=100*abs(PDg-QDg)/QDg;%disp(['��ֵ������',num2str(roundn(re,-2)),'%'])                   %�������� ���������λС��

[max20,position20]=min(LOOEE(position1,:));                                                 %Ѱ������ƽ�����Ӳ��Թ������������ߵ�λ��
Pstorage=DDLS_train;P=DDLS_test(:,test(position20));%Pstorage(:,test(position20))=[];       %ȡ�����޳�������������
Qstorage=f_train;Q0=Qstorage(:,test(position20));%Qstorage(:,test(position20))=[];          %ȡ�����޳���У�������
net=newgrnn(Pstorage,Qstorage,spread(position1));PSD0=net(P);                               %��ô�ʱ�ķ��ݽ��
[top10,position30]=max(PSD0);PDg=dmin+(position30-1)*5*10^-9;                               %Ѱ�ҷ���PSD��ֵλ��
[top20,position40]=max(Q0);QDg=dmin+(position40-1)*5*10^-9;                                 %Ѱ������Q��ֵλ��
re0=100*abs(PDg-QDg)/QDg;%disp(['��ֵ������',num2str(roundn(re,-2)),'%'])                  %�������� ���������λС��

set(0,'defaultfigurecolor','w');
% % subplot(1,2,1);
% plot(spread,LOOCV,'-or','MarkerSize',2.8,'LineWidth',1);
% xlim([smin,smax]);box on;grid on;
% xlabel('ƽ������ֵ(a.u.)','FontName','����','FontSize',10.5);ylabel('����ָ��\itJ_{\Sigma}^{2}(a.u.)','FontName','����','FontSize',10.5);
% title('��ͬƽ�������µķ���Ч������ָ��Ա�ͼ','FontName','����','FontSize',10.5);
subplot(1,2,1);
plot(D*10^9,PSD0,'-or','MarkerSize',2.8,'LineWidth',1);
hold on
plot(D*10^9,Q0,'g','LineWidth',1);
legend({'���������ֲ�','���������ֲ�'},'location','best','FontName','����','FontSize',10.5)
xlim([dmin*10^9,dmax*10^9]);box on;grid on;
xlabel('��������(nm)','FontName','����','FontSize',10.5);ylabel('�����������Ƶ��(a.u.)','FontName','����','FontSize',10.5);
% title('GRNN���ݿ��������ֲ�������Ч��ͼ','FontName','����','FontSize',10.5);

subplot(1,2,2);
plot(D*10^9,PSD,'-or','MarkerSize',2.8,'LineWidth',1);
hold on
plot(D*10^9,Q,'g','LineWidth',1);
legend({'���������ֲ�','���������ֲ�'},'location','best','FontName','����','FontSize',10.5)
xlim([dmin*10^9,dmax*10^9]);box on;grid on;
xlabel('��������(nm)','FontName','����','FontSize',10.5);ylabel('�����������Ƶ��(a.u.)','FontName','����','FontSize',10.5);
% title('GRNN���ݿ��������ֲ������Ч��ͼ','FontName','����','FontSize',10.5);