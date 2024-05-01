
a=61;Dg=linspace(400*10^-9, 700*10^-9, a);	
b=11;delta=linspace(0.05, 0.15, b);           
smin=1*10^-08;smax=10*10^-08;               
c=10;spread=linspace(smin,smax,c);          
d=a*b;test=linspace(1,a*b,d);              
dmin=200*10^-9;dmax=1300*10^-9;             
n=221;D=linspace(dmin,dmax,n);             
R=12;Theta=linspace(3/18*pi,14/18*pi,R);    
np=1.59;nm=1.3316;m=np/nm;                  
lamda=532*10^-9;phi=pi/4;                   
Kb=1.38*10^-23;T=298.15;eta=0.89*10^-3;     
Tpath=5000;t=linspace(20*10^-5,1.0,Tpath);	
time=t';xtime=[time,time.^2,time.^3,time.^4,time.^5,time.^6,time.^7,time.^8,time.^9,time.^10,time.^11,time.^12,time.^13,time.^14,time.^15,time.^16];%拟合截取幂级数最高次项【无噪声用8次，测试4-16次】
noise=0.001;trainmax=01;                    

tic         
for i=1:length(Theta)
    for k=1:n
        x=nm*pi*D(k)/lamda;                     
        u=cos(Theta(i));                        
        nmax=round(2+x+4*x^(1/3));              
        clear pai tao
        pai(1)=1;pai(2)=3*u;                    
        tao(1)=u;tao(2)=3*cos(2*acos(u));       
        
        nmx=round(max(nmax,abs(m.*x))+16);      
        n0=(1:nmax);nu=(n0+0.5);
        px=sqrt(pi*x/2)*besselj(nu,x);          
        p1x=[sin(x),px(1:nmax-1)];              
        chx=-sqrt(pi*x/2)*bessely(nu,x);        
        ch1x=[cos(x),chx(1:nmax-1)];
        gsx=complex(px,chx);gs1x=complex(p1x,ch1x);
        dnx(nmx)=complex(0,0);                  
        
        
        dn=dnx(n0);                             
        an=((dn./m+n0./x).*px-p1x)./((dn./m+n0./x).*gsx-gs1x);
        bn=((m.*dn+n0./x).*px-p1x)./((m.*dn+n0./x).*gsx-gs1x);
        pin=(2*n0+1)./(n0.*(n0+1)).*pai;tin=(2*n0+1)./(n0.*(n0+1)).*tao;
        S1=(an*pin'+bn*tin');S2=(an*tin'+bn*pin');
        I(k)=lamda^2*((abs(S1).^2).*(sin(phi).^2)+(abs(S2).^2).*(cos(phi).^2));
    end
    Fraction(i,:)=I./sum(I);                
end

for cyclea=1:a
    for cycleb=1:b
        y=((10^(-6))./((delta(cycleb)*(sqrt(2*pi))).*D)).*(exp(-(log(D./Dg(cyclea))).^2./(2*(delta(cycleb)^2))));
        fm((cyclea-1)*b+cycleb)=max(y);            
        f(:,(cyclea-1)*b+cycleb)=y.';               
        for i=1:R
            G(i)=10^-6*((Fraction(i,:)*f(:,(cyclea-1)*b+cycleb))^2);
        end
        GR=[];
        for i=1:R 
            T0=16*pi*(nm^2)*Kb*T*((sin(Theta(i)/2))^2)/(3*eta*lamda^2);
            g=1.*exp((-T0.*t)'*(1./D))*diag(Fraction(i,:));     
            GR=cat(1,GR,g);   
        end
        gR=GR*f(:,(cyclea-1)*b+cycleb);             
        for i=1:R
            gg(:,i)=gR(Tpath*(i-1)+1:Tpath*i,1);
            T0=16*pi*(nm^2)*Kb*T*((sin(Theta(i)/2))^2)/(3*eta*lamda^2);
            for train=1:trainmax
                epsilon_train=normrnd(0,1.0,Tpath,1);
                Gacf_train(:,i)=G(i)*(1+noise*epsilon_train+abs(gR(Tpath*(i-1)+1:Tpath*i,1)).^2);%beita=1
                %                 g_train(:,i)=gg(:,i);              %无噪声组作为训练集
                g_train(:,i)=abs((Gacf_train(:,i)./G(i))-1).^(1/2);
                coefficient_train(:,i)=regress(log(g_train(:,i)),xtime);
                DDLS_train(i,(train-1)*a*b+(cyclea-1)*b+cycleb)=-T0/coefficient_train(1,i);
            end
            epsilon=normrnd(0,1.0,Tpath,1);         %产生均值为0，标准差1，方差1的正态分布随机数添加于测试集
            Gacf_test(:,i)=G(i)*(1+noise*epsilon+abs(gR(Tpath*(i-1)+1:Tpath*i,1)).^2);%beita=1
            %             g_test(:,i)=gg(:,i);                   %无噪声组作为测试集
            g_test(:,i)=abs((Gacf_test(:,i)./G(i))-1).^(1/2);
            coefficient_test(:,i)=regress(log(g_test(:,i)),xtime);
            DDLS_test(i,(cyclea-1)*b+cycleb)=-T0/coefficient_test(1,i);
            %DLS(i,(cyclea-1)*b+cycleb)=Fraction(i,:)*f(:,(cyclea-1)*b+cycleb)/(Fraction(i,:)*(f(:,(cyclea-1)*b+cycleb)./D'));
        end
    end
end


f_train=[];
for train=1:trainmax
    f_train=cat(2,f_train,f);
end

tic
for cyclec=1:c
    for cycled=1:d
        Pstorage=DDLS_train;P=DDLS_test(:,test(cycled));%Pstorage(:,test(cycled))=[];       %取出并剔除测试输入向量
        Qstorage=f_train;Q=Qstorage(:,test(cycled));%Qstorage(:,test(cycled))=[];           %取出并剔除检校输出向量
        net=newgrnn(Pstorage,Qstorage,spread(cyclec));PSD=net(P);
        LOOEE(cyclec,cycled)=sum((Q-PSD).^2)/(sum(Q.^2));                                   %J：平方差和/原曲线平方和
        %LOOEE(cyclec,cycled)=sum((Q-PSD).^2)/(fm(cycled)^2);                               %2：平方差和/ft最大值^2
    end
end

LOOCV=sum(LOOEE,2);                                                                         %某一spread值平方差和之和，记录各曲线评价指标数值
toc

[min1,position1]=min(LOOCV);                                                                %寻找最优平滑因子的位置
[max2,position2]=max(LOOEE(position1,:));                                                   %寻找最优平滑因子测试过程中最差曲线的位置

Pstorage=DDLS_train;P=DDLS_test(:,test(position2));%Pstorage(:,test(position2))=[];         %取出并剔除测试输入向量
Qstorage=f_train;Q=Qstorage(:,test(position2));%Qstorage(:,test(position2))=[];             %取出并剔除检校输出向量
net=newgrnn(Pstorage,Qstorage,spread(position1));PSD=net(P);                                %获得此时的反演结果
[top1,position3]=max(PSD);PDg=dmin+(position3-1)*5*10^-9;                                   %寻找反演PSD峰值位置
[top2,position4]=max(Q);QDg=dmin+(position4-1)*5*10^-9;                                     %寻找理论Q峰值位置
re=100*abs(PDg-QDg)/QDg;%disp(['峰值相对误差',num2str(roundn(re,-2)),'%'])                   %求相对误差 输出保留两位小数

[max20,position20]=min(LOOEE(position1,:));                                                 %寻找最优平滑因子测试过程中最优曲线的位置
Pstorage=DDLS_train;P=DDLS_test(:,test(position20));%Pstorage(:,test(position20))=[];       %取出并剔除测试输入向量
Qstorage=f_train;Q0=Qstorage(:,test(position20));%Qstorage(:,test(position20))=[];          %取出并剔除检校输出向量
net=newgrnn(Pstorage,Qstorage,spread(position1));PSD0=net(P);                               %获得此时的反演结果
[top10,position30]=max(PSD0);PDg=dmin+(position30-1)*5*10^-9;                               %寻找反演PSD峰值位置
[top20,position40]=max(Q0);QDg=dmin+(position40-1)*5*10^-9;                                 %寻找理论Q峰值位置
re0=100*abs(PDg-QDg)/QDg;%disp(['峰值相对误差',num2str(roundn(re,-2)),'%'])                  %求相对误差 输出保留两位小数

set(0,'defaultfigurecolor','w');
% % subplot(1,2,1);
% plot(spread,LOOCV,'-or','MarkerSize',2.8,'LineWidth',1);
% xlim([smin,smax]);box on;grid on;
% xlabel('平滑因子值(a.u.)','FontName','宋体','FontSize',10.5);ylabel('评价指标\itJ_{\Sigma}^{2}(a.u.)','FontName','宋体','FontSize',10.5);
% title('不同平滑因子下的反演效果评价指标对比图','FontName','宋体','FontSize',10.5);
subplot(1,2,1);
plot(D*10^9,PSD0,'-or','MarkerSize',2.8,'LineWidth',1);
hold on
plot(D*10^9,Q0,'g','LineWidth',1);
legend({'反演粒径分布','理论粒径分布'},'location','best','FontName','宋体','FontSize',10.5)
xlim([dmin*10^9,dmax*10^9]);box on;grid on;
xlabel('颗粒粒径(nm)','FontName','宋体','FontSize',10.5);ylabel('颗粒粒径体积频度(a.u.)','FontName','宋体','FontSize',10.5);
% title('GRNN反演颗粒粒径分布的最优效果图','FontName','宋体','FontSize',10.5);

subplot(1,2,2);
plot(D*10^9,PSD,'-or','MarkerSize',2.8,'LineWidth',1);
hold on
plot(D*10^9,Q,'g','LineWidth',1);
legend({'反演粒径分布','理论粒径分布'},'location','best','FontName','宋体','FontSize',10.5)
xlim([dmin*10^9,dmax*10^9]);box on;grid on;
xlabel('颗粒粒径(nm)','FontName','宋体','FontSize',10.5);ylabel('颗粒粒径体积频度(a.u.)','FontName','宋体','FontSize',10.5);
% title('GRNN反演颗粒粒径分布的最差效果图','FontName','宋体','FontSize',10.5);