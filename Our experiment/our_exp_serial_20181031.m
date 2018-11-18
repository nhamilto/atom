function [field,field1,field2,field3,meanFields,meanFields1,meanFields2,meanFields3]=our_exp_serial

% speakers
xyzS=[37.8470   11.4078    0.0054
   26.2353   41.0319    0.0176
    1.5697   39.0730   -0.0671
  -37.5346   22.5992   -0.0153
  -41.7272   -0.8058    0.0073
  -27.0696  -37.4858   -0.0080
    0.6702  -45.6852    0.0070
   40.0092  -30.1352    0.0530];
% mics
xyzM=[38.1614   10.7585   -0.0098
   26.8672   40.7957    0.0085
    2.2282   39.1655   -0.0112
  -37.3123   23.2264    0.0298
  -41.8723   -0.1581    0.0020
  -27.7180  -37.1518   -0.0317
   -0.0532  -45.8070   -0.0159
   39.6990  -30.8291    0.0282];
% final 2D aray
xyS = fliplr(xyzS(:,1:2));
xyM = fliplr(xyzM(:,1:2));
xy=[xyS;xyM];
% xy=xy-40;
S=8;
R=8;
%%
% get a file with tt
[file_name,path_name]=uigetfile('*.mat','Pick a file with tt');
if file_name==0
    return
end
%load([path_name,file_name],'tt_out'); % tt_out is filtered signal tt
load([path_name,file_name],'tt'); % tt_out is filtered signal tt
%% compencation for hardware time delays
% delay times in hardware (ms): [i,j] stands for the i-th Speaker and j-th
% Microphone
tDel=[[1.05875, 1.06001389, 1.06030556, 1.06488889, 1.06494444, 1.06097222, 1.06158333, 1.06022222],
    [1.12, 1.12, 1.12, 1.12434722, 1.12206944, 1.125, 1.125, 1.12],
    [1.11554167, 1.11765278, 1.11593056, 1.11819444, 1.11965278, 1.11826389, 1.11522222, 1.11504167],
    [1.04314257, 1.044875, 1.05921592, 1.06113889, 1.05920833, 1.06051389, 1.060875, 1.04394444],
    [1.11075, 1.11193056, 1.11, 1.11398611, 1.11497222, 1.11497222, 1.11405556, 1.11004167],
    [1.0805, 1.08470833, 1.085125, 1.08493056, 1.08505556, 1.08504167,1.08465278, 1.07798611],
    [1.13984722, 1.13931944, 1.13798611, 1.14, 1.14004167, 1.13959722, 1.139875, 1.13877778],
    [1.09001389, 1.0905, 1.10404167, 1.09431944, 1.09194444, 1.09, 1.09, 1.08895833]];
delT=tDel.';
delT=delT(:);
[nrays,Ntime]=size(tt); %#ok<NODEF>
% tt=tt-repmat(delT,1,Ntime);
tt=tt*1e-3; % convert in s
%%
%---------------------------------------
% estimation of the mean fields
meanField=[];
meanField1=[];
meanFields3=[];
N_frames=2;
% mean_frames=3;
% mean_frames=2*N_frames+1; % used for the mean field reconstruction
mean_frames=Ntime;
if Ntime/mean_frames~=fix(Ntime/mean_frames)
    error('Ntime=%g, mean_frames=%g, Ntime/mean_frames must be integer',Ntime,mean_frames);
end
for i=1:Ntime/mean_frames
    tmp=spatialMean_serial(S,R,xy,tt(:,(i-1)*mean_frames+1:i*mean_frames),320,360,0,0);
    meanField=[meanField tmp]; %#ok<AGROW>
    [tmp,delays]=spatialMean_serial(S,R,xy,tt(:,(i-1)*mean_frames+1:i*mean_frames),320,360,1,0);
%     [tmp,delays]=spatialMean_serial_maxD(S,R,xy,tt(:,(i-1)*mean_frames+1:i*mean_frames),320,360,1,15,0);
    meanField1=[meanField1 tmp]; %#ok<AGROW>
%     [tmp,delays3]=spatialMean_serial_t(S,R,xy,tt(:,(i-1)*mean_frames+1:i*mean_frames),320,360,0);
%     meanField3=[meanFields3 tmp]; %#ok<AGROW>
    figure(i);
    plot(delays*1000,'o','linewidth',2,'color',[0 0.5 0],'MarkerSize',10)
    xlabel('Path index','fontsize',12,'fontweight','bold')
    ylabel('Time delay (ms)','fontsize',12,'fontweight','bold')
    
end
tmp=input('i0  and Ndelays: ');
i0=tmp(1);
Ndel=tmp(2);
t0=i0*mean_frames-round((mean_frames-1)/2);
% t0=round(Ntime/2);
frames=t0-N_frames:t0+N_frames;
meanFields=[meanField(t0) meanField(frames)];
meanFields1=[meanField1(t0) meanField1(frames)];
disp('%----------------------------------------')
disp(meanFields(1));
disp('%----------------------------------------')
disp(meanFields1(1));
disp('%----------------------------------------')
if Ndel~=0 % delays numbers are indicated
    delaysFlg=1;
else
    delaysFlg=0; % no delays are indicated
end
[tmp,d2]=spatialMean_serial_maxD(S,R,xy,tt(:,(i0-1)*mean_frames+1:i0*mean_frames),320,360,delaysFlg,Ndel,0);
meanFields2=[tmp(mean_frames-N_frames) tmp];
disp(meanFields2(1));
disp('%----------------------------------------')
figure(i0)
hold on
plot(d2*1000,'rs','linewidth',2,'MarkerSize',10);

% reconstruction of the fluctuations
Lx=[min(xy(:,1)) max(xy(:,1))];
Ly=[min(xy(:,2)) max(xy(:,2))];
xv=linspace(Lx(1),Lx(2),65);
yv=linspace(Ly(1),Ly(2),65);
% sigma.T=0.04;
% sigma.vx=0.04;
% sigma.vy=0.04;
% sigma.T=0.2;
% sigma.vx=0.3;
% sigma.vy=0.3;
sigma.T=meanFields(1).std_dT;
sigma.vx=meanFields(1).std_dvx;
sigma.vy=meanFields(1).std_dvy;

sigma.n=1e-5;
sigma.x=1e-3;
lT=15;
lv=15;
%----------------------------------------
U.x=[meanFields.vx]';
U.y=[meanFields.vy]';
% sigma.vx=std(U.x);
% sigma.vy=std(U.y);
U.sig=sqrt((sigma.vx^2+sigma.vy^2)/2);
% sigma.T=0.5*U.sig;
U1.x=[meanFields1.vx]';
U1.y=[meanFields1.vy]';
U1.sig=U.sig;
sigma1=sigma;
sigma1.T=meanFields1(1).std_dT;
sigma1.vx=meanFields1(1).std_dvx;
sigma1.vy=meanFields1(1).std_dvy;

U2.x=[meanFields2.vx]';
U2.y=[meanFields2.vy]';
U2.sig=U.sig;
sigma2=sigma;
sigma2.T=meanFields2(1).std_dT;
sigma2.vx=meanFields2(1).std_dvx;
sigma2.vy=meanFields2(1).std_dvy;

%-------------------------------------------
tau=0.5;
funType='gauss';
estFrame='single';
SI=0;
Cnstr=[];
InvType='Full';
interp=0;
showFig=1;

% [field,~,~,R_dd0, R_md]=t_stochastic_inverse2...
%     ('n',[],'n',[],S,R,Lx,Ly,xv,yv,sigma,lT,lv,...
%     meanFields,U,tau,t0,frames,funType,estFrame,SI,Cnstr,InvType,interp,showFig);

field1=t_stochastic_inverse2...
    ('n',[],'n',[],S,R,Lx,Ly,xv,yv,sigma1,lT,lv,...
    meanFields1,U1,tau,t0,frames,funType,estFrame,SI,Cnstr,InvType,interp,showFig);

field2=t_stochastic_inverse2...
    ('n',[],'n',[],S,R,Lx,Ly,xv,yv,sigma2,lT,lv,...
    meanFields2,U2,tau,t0,frames,funType,estFrame,SI,Cnstr,InvType,interp,showFig);

