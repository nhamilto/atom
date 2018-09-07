function [meanField,delays]=spatialMean_serial_seq(S,R,xy,tt,Lo,Hi,delaysFlg,showFig)
% spatialMean_serial does the same job as spatialMean (see its
% description). The difference is that the input tt is a matrix of
% size [S*R, time] rather than a vector [S*R,1]. All travel times measured
% at different times are used for the reconstruction of mean T and V
% fields. This allows serial data acquisition experiments when only one
% source a time is used. The output meanField is a struct vector of length
% time rather than a struct variable.

[~, Ntime]=size(tt);
d=[];G=[];l0=0;
index=zeros(Ntime+1,1);
Li=cell(Ntime,1);t=Li;in=Li;
ind=[];
for i=1:Ntime
    [Li{i},t{i},in{i},theta]=ttFiltr(S,R,xy,tt(:,i),Lo,Hi,showFig);
    d=[d;t{i}./Li{i}];
    l=length(Li{i});
    G(l0+1:l0+l,1:2)=-theta;
    if delaysFlg
        for j=1:l
            G(l0+j,in{i}(j)+2)=1/Li{i}(j);
        end
    end
    l0=size(G,1);
    index(i+1)=length(d);
end
lLi=index(Ntime+1);
G=[ones(lLi,1) G];
G0=G;
ldel=S*R;
ind0=1:ldel;
delays=zeros(ldel,1);
if delaysFlg
    for i=1:ldel
        G=G0;
        ind=ind0;
        ind(i)=[];
        G(:,ind+3)=[];
        G_1=pinv(G);
        aver=G_1*d;
        delays(i)=aver(4);
    end
end
G=G0(:,1:3);
for i=1:Ntime
    d(index(i)+1:index(i+1))=d(index(i)+1:index(i+1))-delays./Li{i};
end
G_1=pinv(G);
aver=G_1*d;
er=d-G*aver;
sig2=er'*er/(lLi-size(G,2));
std_a=sqrt(sig2*diag(G_1*G_1'));
c0est=1/aver(1);
c02=c0est*c0est;
dc=c02*std_a(1);
dc2=2*c0est*dc;
T0est=c02/343/343*293;
dT=dc2/343/343*293;
vx0est=aver(2)*c02;
dvx=c02*std_a(2);%+aver(2)*dc2;
vy0est=aver(3)*c02;
dvy=c02*std_a(3);%+aver(3)*dc2;
for i=1:Ntime
    dtt=Li{i}.*er(index(i)+1:index(i+1));
    meanField(i).c=c0est;
    %meanField.dc=dc;
    meanField(i).T=T0est;
    %meanField.dT=dT;
    meanField(i).vx=vx0est;
    %meanField.dvx=dvx;
    meanField(i).vy=vy0est;
    %meanField.dvy=dvy;
    
    meanField(i).std_dc=dc;
    meanField(i).std_dT=dT;
    meanField(i).std_dvx=dvx;
    meanField(i).std_dvy=dvy;
    %%%%%%%%%%%%%%%%%%%%%%%%% misc
    % meanField.std_tt=sig_tt;
    % meanField.std_x=sig_x;
    meanField(i).dtt=dtt;
    meanField(i).tt=t{i};
    meanField(i).index=in{i};
    meanField(i).data=[-c0est*c0est*dtt in{i}];
    meanField(i).xy=xy;
    meanField(i).Li=Li{i};
end