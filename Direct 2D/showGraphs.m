function showGraphs(menuName,S,R,meanFields,xv,yv,fluct,interp,SingValG,SingValC,cMapName)
% showGraphs shows graphs and makes movies of the 2D acoustical travel time
% tomography problem.
% Can be launched without any input arguments. In this case it will offer
% to choose a file with the results (for example, it can be the file with
% the results of the inverse problem created by 't_stochastic_inverse2.m').
% If launched with specified parameters, the syntax is:
%
% showGraphs(menuName,S,R,meanFields,xv,yv,fluct,interp,SingValG,SingValC);
%
% Inputs:
% menuName - a caption of the graphic menu, string; can be empty:
% menuName=[];
% S - number of sound sources, integer;
% R - number of receivers, integer;
% meanFields - a struct array of size [1,time] with fields:
%   meanFields.T - the mean value of T field;
%   meanFields.vx - the mean value of vx field;
%   meanFields.vy - the mean value of vy field;
%   Optional fields:
%       meanFields.t - time in seconds to label the time-axis;
%       meanFields.tt - travel times, a vector of the length less or equal
%           S*R (can be less if some of the travel times are filtered out);
%       meanFields.dtt - travel times due to fluctuations, a vector of the
%           same length as meanFields.tt;
%       meanFields.xy - the coordinates of sources and receivers;
%       meanFields.index - the indeces of the valid travel times;
%       meanFields.std_dT - the standard deviation of the errors in T_0;
%       meanFields.std_dvx - the standard deviation of the errors in vx_0;
%       meanFields.std_dvy - the standard deviation of the errors in vy_0;
% xv - a vector of spatial points along x axis where the fields are given;
% yv - a vector of spatial points along y axis where the fields are given;
% fluct - a struct variable with fields:
%   fluct.dT - fluctuations of T, a matrix (x,y,time);
%   fluct.dC - fluctuations of c, a matrix (x,y,time);
%   fluct.vx - fluctuations of vx, a matrix (x,y,time);
%   fluct.vy - fluctuations of vy, a matrix (x,y,time);
%   fluct.dTExpNMSE - NMSE of T, a matrix (x,y,time);
%   fluct.dTExpAverNMSE - spatial average of NMSE of T, a scalar;
%   fluct.dTExpSTD - STD of T, a matrix (x,y,time);
%   fluct.dTExpAverSTD - spatial average of STD of T, a scalar;
%   fluct.vxExpNMSE - NMSE of vx, a matrix (x,y,time);
%   fluct.vxExpAverNMSE - spatial average of NMSE of vx, a scalar;
%   fluct.vxExpSTD - STD of vx, a matrix (x,y,time);
%   fluct.vxExpAverSTD - spatial average of STD of vx, a scalar;
%   fluct.vyExpNMSE - NMSE of vy, a matrix (x,y,time);
%   fluct.vyExpAverNMSE - spatial average of NMSE of vy, a scalar;
%   fluct.vyExpSTD - STD of vy, a matrix (x,y,time);
%   fluct.vyExpAverSTD - spatial average of STD of vy, a scalar;
% A movie (".avi" - file) will be created if time > 1.
% interp - interpolation of the displayed fields;
%       interp = 0 - no interpolation; interp = 1 - one extra point will be added in the middle
%       of two adjacent points in space and time, etc. (2^interp)
% SingValG - a vector of singular values of a matrix being inverted (e.g.,
%   the forward problem matrix G or Rdd);
% SingValC - another vector of singular values of a matrix being inverted (e.g.,
%   a matrix of constraints C);
% cMapName - a name of a standard colormap to use for images, e.g. 'jet' or 'parula'; 
% if not specified, 'jet' is used. 


if nargin==0
    cMapName='jet';
    [FileName,PathName] = uigetfile('*.mat','Select a MAT-file');
    if FileName~=0
        load([PathName,FileName],'meanFields','t0','S','R','xy','xv','yv','field','fields',...
            'SingValG','SingValC','SI','Cnstr_flag');
        menuName=sprintf('Fields at t=%d',t0);
        prompt={'Enter the interpolation factor:'};
        def={'0'};
        dlgTitle='Input for interpolation';
        lineNo=1;
        answer=inputdlg(prompt,dlgTitle,lineNo,def);
        interp=str2double(answer{1});
        showGraphs(menuName,S,R,meanFields(1),xv,yv,field,interp,SingValG,SingValC,cMapName);
        if exist('fields','var') && ~isempty(fields)
            menuName='Fields at specified frames';
            showGraphs(menuName,S,R,meanFields(2:end),xv,yv,fields,interp,SingValG,SingValC,cMapName);
        end
        if SI==1
            load([PathName,FileName],'field_SI','fields_SI');
            menuName=sprintf('SI Fields at t=%d',t0);
            showGraphs(menuName,S,R,meanFields(1),xv,yv,field_SI,interp,SingValG,SingValC);
            if ~isempty(fields_SI)
                menuName='SI Fields at specified frames';
                showGraphs(menuName,S,R,meanFields(2:end),xv,yv,fields_SI,interp,SingValG,SingValC);
            end
        end
        if Cnstr_flag==1
            load([PathName,FileName],'field_Cnstr','fields_Cnstr');
            menuName=sprintf('Cnstr Fields at t=%d',t0);
            showGraphs(menuName,S,R,meanFields(1),xv,yv,field_Cnstr,interp,SingValG,SingValC);
            if ~isempty(fields_Cnstr)
                menuName='Cnstr Fields at specified frames';
                showGraphs(menuName,S,R,meanFields(2:end),xv,yv,fields_Cnstr,interp,SingValG,SingValC);
            end
        end
        
        return
    else
        warndlg('No data available!','Data info');
        return
    end
end
Choice=1;
showFlow=0;
if ~exist('cMapName','var') || isempty(cMapName)
    cMapName='jet';
end

if ~exist('menuName','var') || isempty(menuName)
    menuName='Choose a plot';
end
if ~exist('interp','var') || isempty(interp)
    interp=0;
end
if ~exist('meanFields','var') || isempty(meanFields)
    VX=fluct.dT;
    [M,grids,Ntime]=size(VX);
    for i=1:Ntime
        meanFields(i).T=273.15;
        meanFields(i).vx=0;
        meanFields(i).vy=0;
        meanFields(i).std_dT=0;
        meanFields(i).std_dvx=0;
        meanFields(i).std_dvy=0;
    end
end

while Choice~=17
    Choice=menu(menuName,...
        'Array scheme','Mean fields',...
        'T field','u field', 'v field', 'Abs. wind field',...
        'Add wind flow','Travel times','Travel times differences','Angles',...
        'G spectrum','C spectrum','dT NMSE & STD','u NMSE & STD','v NMSE & STD','V STD','Exit');
    switch Choice
        case 1 % Array
            if exist('meanFields','var') & isfield(meanFields,'xy') & ~isempty(meanFields(1).xy)
                xy=meanFields(1).xy;
                %                 Lx=[min([xv(1);xy(:,1)]) max([xv(end);xy(:,1)])];
                %                 Ly=[min([yv(1);xy(:,2)]) max([yv(end);xy(:,2)])];
                Lx=round([min(xy(:,1)) max(xy(:,1))]);
                Ly=round([min(xy(:,2)) max(xy(:,2))]);
                figure;
                plot(Lx,[Ly(1),Ly(1)],'k','linewidth',2);
                dLx=Lx(2)-Lx(1);
                dLy=Ly(2)-Ly(1);
                axis equal
                set(gca,'XLim',[Lx(1)-dLx/6, Lx(2)+dLx/6],'YLim',[Ly(1)-dLy/6, Ly(2)+dLy/6])%,...
                hold on;
                plot([Lx(2),Lx(2)],Ly,'k','linewidth',2);
                plot(Lx,[Ly(2),Ly(2)],'k','linewidth',2);
                plot([Lx(1),Lx(1)],Ly,'k','linewidth',2);
                xlabel('x (m)','FontWeight','Bold','FontSize',12);
                ylabel('y (m)','FontWeight','Bold','FontSize',12);
                h3=[];h4=[];
                Ntime=length(meanFields);
                if Ntime>1
                    aviobj = avifile('Array.avi','compression', 'i420','quality', 100,'fps',6);
                end
                for t=1:Ntime
                    for i=1:S
                        for j=1:R
                            num=(i-1)*R+j;
                            if any(ismember(meanFields(t).index,num))
                                h3=plot([xy(i,1) xy(j+S,1)],[xy(i,2) xy(j+S,2)],'g');%'Color',[0 0.7 0]);
                            else
                                h4=plot([xy(i,1) xy(j+S,1)],[xy(i,2) xy(j+S,2)],'--','Color',[170 170 170]/255);
                            end
                        end
                    end
                    h1=plot(xy(1:S,1),xy(1:S,2),'or','markersize',10,'MarkerFaceColor','r','linewidth',2);
                    h2=plot(xy(S+1:R+S,1),xy(S+1:R+S,2),'sb','markersize',10,'MarkerFaceColor','b','linewidth',2);
                    legend([h1,h2],'Sources','Receivers',-1);
                    if ~isempty(h3) & isempty(h4)
                        legend([h1,h2,h3],'Sources','Receivers','Rays',-1);
                    end
                    if ~isempty(h3) & ~isempty(h4)
                        legend([h1,h2,h3,h4],'Sources','Receivers','Rays','Omitted rays',-1);
                        title(sprintf('Total number of valid rays: %g',length(meanFields(t).index)),...
                            'fontsize',14,'fontweight','bold');
                    end
                    for i=1:S
                        str=sprintf('S%d',i);
                        text(xy(i,1), xy(i,2), str,'FontWeight','Bold','FontSize',12);
                    end
                    for i=1:R
                        str=sprintf('R%d',i);
                        text(xy(i+S,1), xy(i+S,2), str,'FontWeight','Bold','FontSize',12);
                    end
                    if Ntime>1
                        frame = getframe(gcf);
                        aviobj = addframe(aviobj,frame);
                    end
                end
                if Ntime>1
                    aviobj = close(aviobj);
                end
            else
                warndlg('No data available!','Fig. info');
            end
            
        case 2 % Mean fields
            
            if exist('meanFields','var') & isfield(meanFields,'T') & isfield(meanFields,'vy') & isfield(meanFields,'vx')
                T0=[meanFields.T]'-273.15;
                vx0=[meanFields.vx]';
                vy0=[meanFields.vy]';
                Ntime=length(T0);
                if isfield(meanFields,'std_dT')
                    std_T0=[meanFields.std_dT]';
                else
                    std_T0=zeros(Ntime,1);
                end
                if isfield(meanFields,'std_dvx')
                    std_vx0=[meanFields.std_dvx]';
                else
                    std_vx0=zeros(Ntime,1);
                end
                if isfield(meanFields,'std_dvy')
                    std_vy0=[meanFields.std_dvy]';
                else
                    std_vy0=zeros(Ntime,1);
                end
                if Ntime>1 && interp>0
                    step=1/2^interp;
                    t_ax=[0:step:Ntime-1]';
                    T0=interp1([0:Ntime-1],T0,t_ax);
                    vx0=interp1([0:Ntime-1],vx0,t_ax);
                    vy0=interp1([0:Ntime-1],vy0,t_ax);
                    std_T0=interp1([0:Ntime-1],std_T0,t_ax);
                    std_vx0=interp1([0:Ntime-1],std_vx0,t_ax);
                    std_vy0=interp1([0:Ntime-1],std_vy0,t_ax);
                    Ntime=length(T0);
                end
                if ~isfield(meanFields(1),'t') || isempty(meanFields(1).t)
                    timeaxis=(0:Ntime-1)';
                    xunit=' (samples)';
                else
                    timeaxis=[meanFields(1).t]';
                    xunit=' (s)';
                end
                %                  matlab_version=version;
                figure;
                if size(T0,1)==1
                    plot(timeaxis,T0,'p','linewidth',2);
                    if ~all(std_T0==0)
                        hold on
                        plot([timeaxis timeaxis],[T0-std_T0 T0+std_T0],'r')
                    end
                    %                     if matlab_version(1)=='7'
                    %                         h=errorbar('v6',timeaxis,T0,std_T0,'p');
                    %                     else
                    %                         h=errorbar(timeaxis,T0,std_T0,'p');
                    %                         %
                    %                     end
                else
                    %                     if matlab_version(1)=='7'
                    %                         h=errorbar('v6',timeaxis,T0,std_T0);
                    %                     else
                    %                         h=errorbar(timeaxis,T0,std_T0);
                    %                     end
                    
                    
                    if ~all(std_T0==0)
                        timeaxis1=timeaxis(end:-1:1);
                        data=T0+std_T0;
                        data=data(end:-1:1);
                        fill([timeaxis;timeaxis1],[T0-std_T0;data],[1 0.6 1]);
                        hold on
                        plot(timeaxis,T0-std_T0,'r',timeaxis,T0+std_T0,'r')
                    end
                    plot(timeaxis,T0,'linewidth',2)
                    grid
                    set(gca,'Layer','top')
                end
                axis tight
                %                 if all(std_T0==0)
                %                     delete(h(1));
                %                 else
                %                     set(h(1),'Color','r');
                %                 end
                %                 set(h(2),'linewidth',2)
                ylabel('T_0 (^oC)','fontsize',12,'fontweight','bold');
                xlabel(['Time',xunit],'fontsize',12,'fontweight','bold');
                figure;
                if size(vx0,1)==1
                    plot(timeaxis,vx0,'p','linewidth',2);
                    if ~all(std_vx0==0)
                        hold on
                        plot([timeaxis timeaxis],[vx0-std_vx0 vx0+std_vx0],'r')
                    end
                else
                    if ~all(std_vx0==0)
                        timeaxis1=timeaxis(end:-1:1);
                        data=vx0+std_vx0;
                        data=data(end:-1:1);
                        fill([timeaxis;timeaxis1],[vx0-std_vx0;data],[1 0.6 1]);
                        hold on
                        plot(timeaxis,vx0-std_vx0,'r',timeaxis,vx0+std_vx0,'r')
                    end
                    plot(timeaxis,vx0,'linewidth',2)
                    grid
                    set(gca,'Layer','top')
                end
                axis tight
                %                 if all(std_vx0==0)
                %                     delete(h(1));
                %                 else
                %                     set(h(1),'Color','r');
                %                 end
                %                 set(h(2),'linewidth',2)
                ylabel('v_{x0} (m/s)','fontsize',12,'fontweight','bold');
                xlabel(['Time',xunit],'fontsize',12,'fontweight','bold');
                figure;
                if size(vy0,1)==1
                    plot(timeaxis,vy0,'p','linewidth',2);
                    if ~all(std_vy0==0)
                        hold on
                        plot([timeaxis timeaxis],[vy0-std_vy0 vy0+std_vy0],'r')
                    end
                else
                    if ~all(std_vy0==0)
                        timeaxis1=timeaxis(end:-1:1);
                        data=vy0+std_vy0;
                        data=data(end:-1:1);
                        fill([timeaxis;timeaxis1],[vy0-std_vy0;data],[1 0.6 1]);
                        hold on
                        plot(timeaxis,vy0-std_vy0,'r',timeaxis,vy0+std_vy0,'r')
                    end
                    plot(timeaxis,vy0,'linewidth',2)
                    grid
                    set(gca,'Layer','top')
                end
                axis tight
                %                 if all(std_vy0==0)
                %                     delete(h(1));
                %                 else
                %                     set(h(1),'Color','r');
                %                 end
                %                 set(h(2),'linewidth',2)
                ylabel('v_{y0} (m/s)','fontsize',12,'fontweight','bold');
                xlabel(['Time',xunit],'fontsize',12,'fontweight','bold');
            else
                warndlg('No data available!','Fig. info');
            end
            
        case 3 % Temperature field
            if exist('fluct','var') && isfield(fluct,'dT')
                VX=fluct.dT;
                [M,grids,Ntime]=size(VX);
                if exist('meanFields','var') && isfield(meanFields,'T')
                    for i=1:Ntime
                        VX(:,:,i)=VX(:,:,i)+meanFields(i).T-273.15;
                    end
                end
                if interp>0
                    VX=interpn(VX,interp);
                    Ntime=size(VX,3);
                end
                v=VX(:,:,1);
                fig=figure;
                set(fig,'DoubleBuffer','on');
                h1=imagesc([xv(1),xv(end)],[yv(1),yv(end)],v');
                colormap(cMapName);
                axis equal;
                set(gca,'Xlim',[xv(1),xv(end)],'Ylim',[yv(1),yv(end)],'YDir','normal','nextplot','replace','Visible','on');
                title('T field','FontWeight','Bold','FontSize',12);
                xlabel('x (m)','FontWeight','Bold','FontSize',12);
                ylabel('y (m)','FontWeight','Bold','FontSize',12);
                Cmin=min(min(min(VX)));
                Cmax=max(max(max(VX)));
                if Cmax>Cmin
                    h2=get(h1,'Parent');
                    set(h2,'CLim',[Cmin Cmax]);
                end
                if showFlow>0
                    V1=fluct.vx;
                    V2=fluct.vy;
                    Ntime1=size(V1,3);
                    if exist('meanFields','var') && isfield(meanFields,'vx')
                        for i=1:Ntime1
                            V1(:,:,i)=V1(:,:,i)+meanFields(i).vx;
                            V2(:,:,i)=V2(:,:,i)+meanFields(i).vy;
                        end
                    end
                    xv1=xv;yv1=yv;
                    if interp>0
                        V1=interpn(V1,interp);
                        V2=interpn(V2,interp);
                        xv1=linspace(xv(1),xv(end),size(V1,1));
                        yv1=linspace(yv(1),yv(end),size(V1,2));
                    end
                    [Xv,Yv]=meshgrid(xv1,yv1);
                end
                hc=colorbar;
                set(hc,'fontweight','bold');
                ht=get(hc,'Title');
                set(ht,'String','^oC','fontweight','bold','fontsize',12);
                if Ntime>1
                    aviobj = VideoWriter('T.avi'); %#ok<TNMLP>
                    aviobj.Quality=50;
                    open(aviobj);
                end
                for i=1:Ntime
                    v=VX(:,:,i);
                    set(h1,'CData',v');
                    if showFlow>0
                        hstr=streamslice(Xv,Yv,V1(:,:,i)',V2(:,:,i)',0.3);
                        set(hstr,'color','k');
                    end
                    if Ntime>1
                        frame = getframe(gcf);
                        writeVideo(aviobj,frame);
                    end
                    if showFlow>0 && i~=Ntime
                        delete(hstr);
                    end
                end
                if Ntime>1
                    close(aviobj);
                end
                showFlow=showFlow-1;
            else
                warndlg('No data available!','Fig. info');
            end
            
        case 4 % vx field
            if exist('fluct','var') && isfield(fluct,'vx')
                VX=fluct.vx;
                [M,grids,Ntime]=size(VX);
                if  exist('meanFields','var') && isfield(meanFields,'vx')
                    for i=1:Ntime
                        VX(:,:,i)=VX(:,:,i)+meanFields(i).vx;
                    end
                end
                if interp>0
                    VX=interpn(VX,interp);
                    Ntime=size(VX,3);
                end
                v=VX(:,:,1);
                fig=figure;
                set(fig,'DoubleBuffer','on');
                h1=imagesc([xv(1),xv(end)],[yv(1),yv(end)],v');
                colormap(cMapName);
                axis equal;
                set(gca,'Xlim',[xv(1),xv(end)],'Ylim',[yv(1),yv(end)],'YDir','normal','nextplot','replace','Visible','on');
                title('v_{x} field','FontWeight','Bold','FontSize',12);
                xlabel('x (m)','FontWeight','Bold','FontSize',12);
                ylabel('y (m)','FontWeight','Bold','FontSize',12);
                Cmin=min(min(min(VX)));
                Cmax=max(max(max(VX)));
                if Cmax>Cmin
                    h2=get(h1,'Parent');
                    set(h2,'CLim',[Cmin Cmax]);
                end
                hc=colorbar;
                set(hc,'fontweight','bold');
                ht=get(hc,'Title');
                set(ht,'String','m/s','fontweight','bold','fontsize',12);
                if Ntime>1
                    aviobj = VideoWriter('vx.avi'); %#ok<TNMLP>
                    aviobj.Quality=50;
                    open(aviobj);
                end
                for i=1:Ntime
                    v=VX(:,:,i);
                    set(h1,'CData',v');
                    if Ntime>1
                        frame = getframe(gcf);
                        writeVideo(aviobj,frame);
                    end
                end
                if Ntime>1
                    close(aviobj);
                end
            else
                warndlg('No data available!','Fig. info');
            end
            
        case 5 % vy field
            if exist('fluct','var') && isfield(fluct,'vy')
                VX=fluct.vy;
                [M,grids,Ntime]=size(VX);
                if exist('meanFields','var') & isfield(meanFields,'vy')
                    for i=1:Ntime
                        VX(:,:,i)=VX(:,:,i)+meanFields(i).vy;
                    end
                end
                if interp>0
                    VX=interpn(VX,interp);
                    Ntime=size(VX,3);
                end
                v=VX(:,:,1);
                fig=figure;
                set(fig,'DoubleBuffer','on');
                h1=imagesc([xv(1),xv(end)],[yv(1),yv(end)],v');
                colormap(cMapName);
                axis equal;
                set(gca,'Xlim',[xv(1),xv(end)],'Ylim',[yv(1),yv(end)],'YDir','normal','nextplot','replace','Visible','on');
                title('v_{y} field','FontWeight','Bold','FontSize',12);
                xlabel('x (m)','FontWeight','Bold','FontSize',12);
                ylabel('y (m)','FontWeight','Bold','FontSize',12);
                Cmin=min(min(min(VX)));
                Cmax=max(max(max(VX)));
                if Cmax>Cmin
                    h2=get(h1,'Parent');
                    set(h2,'CLim',[Cmin Cmax]);
                end
                hc=colorbar;
                set(hc,'fontweight','bold');
                ht=get(hc,'Title');
                set(ht,'String','m/s','fontweight','bold','fontsize',12);
                if Ntime>1
                    aviobj = VideoWriter('vy.avi'); %#ok<TNMLP>
                    aviobj.Quality=50;
                    open(aviobj);
                end
                for i=1:Ntime
                    v=VX(:,:,i);
                    set(h1,'CData',v');
                    if Ntime>1
                        frame = getframe(gcf);
                        writeVideo(aviobj,frame);
                    end
                end
                if Ntime>1
                    close(aviobj);
                end
            else
                warndlg('No data available!','Fig. info');
            end
            
        case 6 % V field
            if exist('fluct','var') && isfield(fluct,'vx') && isfield(fluct,'vy')
                V1=fluct.vx;
                V2=fluct.vy;
                [M,grids,Ntime]=size(V1);
                if exist('meanFields','var')&& isfield(meanFields,'vx') && isfield(meanFields,'vy')
                    for i=1:Ntime
                        V1(:,:,i)=V1(:,:,i)+meanFields(i).vx;
                        V2(:,:,i)=V2(:,:,i)+meanFields(i).vy;
                    end
                end
                if interp>0
                    V1=interpn(V1,interp);
                    V2=interpn(V2,interp);
                    Ntime=size(V1,3);
                end
                VX=sqrt(V1.^2+V2.^2);
                v=VX(:,:,1);
                fig=figure;
                set(fig,'DoubleBuffer','on');
                h1=imagesc([xv(1),xv(end)],[yv(1),yv(end)],v');
                colormap(cMapName);
                axis equal;
                set(gca,'Xlim',[xv(1),xv(end)],'Ylim',[yv(1),yv(end)],'YDir','normal','nextplot','replace','Visible','on');
                title('V=(v_x^2+v_y^2)^{1/2}','FontWeight','Bold','FontSize',12);
                xlabel('x (m)','FontWeight','Bold','FontSize',12);
                ylabel('y (m)','FontWeight','Bold','FontSize',12);
                Cmin=min(min(min(VX)));
                Cmax=max(max(max(VX)));
                if Cmax>Cmin
                    h2=get(h1,'Parent');
                    set(h2,'CLim',[Cmin Cmax]);
                end
                if showFlow>0
                    xv1=xv;yv1=yv;
                    if interp>0
                        xv1=linspace(xv(1),xv(end),size(V1,1));
                        yv1=linspace(yv(1),yv(end),size(V1,2));
                    end
                    [Xv,Yv]=meshgrid(xv1,yv1);
                end
                hc=colorbar;
                set(hc,'fontweight','bold');
                ht=get(hc,'Title');
                set(ht,'String','m/s','fontweight','bold','fontsize',12);
                if Ntime>1
                    aviobj = VideoWriter('V.avi'); %#ok<TNMLP>
                    aviobj.Quality=50;
                    open(aviobj);
                end
                for i=1:Ntime
                    v=VX(:,:,i);
                    set(h1,'CData',v');
                    if showFlow>0
                        hstr=streamslice(Xv,Yv,V1(:,:,i)',V2(:,:,i)',0.3);
                        set(hstr,'color','k');
                    end
                    if Ntime>1
                        frame = getframe(gcf);
                        writeVideo(aviobj,frame);
                    end
                    if showFlow>0 & i~=Ntime
                        delete(hstr);
                    end
                end
                if Ntime>1
                    close(aviobj);
                end
                showFlow=showFlow-1;
            else
                warndlg('No data available!','Fig. info');
            end
            
        case 7 % Add wind flow to current axes
            if exist('fluct','var')&isfield(fluct,'vx')&isfield(fluct,'vy')
                h=findobj;
                if length(h)==1
                    warndlg('Wind flow will be added to T and V fields');
                    showFlow=2;
                else
                    V1=fluct.vx;
                    [M,grids,Ntime]=size(V1);
                    if Ntime>1
                        warndlg('Unable to add flow to the current figure since the fields have several frames.');
                    else
                        V2=fluct.vy;
                        if exist('meanFields','var')&isfield(meanFields,'vx')&isfield(meanFields,'vy')
                            V1=V1+meanFields.vx;
                            V2=V2+meanFields.vy;
                        end
                        xv1=xv;yv1=yv;
                        if interp>0
                            V1=interpn(V1,interp);
                            V2=interpn(V2,interp);
                            xv1=linspace(xv(1),xv(end),size(V1,1));
                            yv1=linspace(yv(1),yv(end),size(V1,2));
                        end
                        [Xv,Yv]=meshgrid(xv1,yv1);
                        hstr=streamslice(Xv,Yv,V1',V2',0.3);
                        set(hstr,'color','k');
                    end
                end
            else
                warndlg('No data available!','Fig. info');
            end
        case 8 % travel times
            if exist('meanFields','var') & isfield(meanFields,'tt') & ~isempty([meanFields.tt])
                tt1=[meanFields.tt];
                tt=1000*tt1;
                Ntime=size(tt,2);
                ti=tt(:,1);
                figure;
                h1=plot(ti,'o:','linewidth',2,'MarkerFaceColor','m','MarkerEdgeColor','m');
                grid
                Cmin=min(min(tt));
                Cmax=max(max(tt));
                if Cmax>Cmin
                    h2=get(h1,'Parent');
                    set(h2,'YLim',[Cmin Cmax]);
                end
                title('Travel times','FontWeight','Bold','FontSize',12);
                xlabel('Index of travel path','FontWeight','Bold','FontSize',12);
                ylabel('t (ms)','FontWeight','Bold','FontSize',12);
                if Ntime>1
                    aviobj = VideoWriter('tt.avi'); %#ok<TNMLP>
                    aviobj.Quality=50;
                    open(aviobj);
                end
                for i=1:Ntime
                    ti=tt(:,i);
                    set(h1,'YData',ti);
                    if Ntime>1
                        frame = getframe(gcf);
                        writeVideo(aviobj,frame);
                    end
                end
                if Ntime>1
                    close(aviobj);
                end
            else
                warndlg('No data available!','Fig. info');
            end
        case 9 % travel times due to fluctuations
            if exist('meanFields','var') & isfield(meanFields,'dtt') & ~isempty([meanFields.dtt])
                dtt1=[meanFields.dtt];
                dtt=1000*dtt1;
                Ntime=size(dtt,2);
                ti=dtt(:,1);
                figure;
                h1=plot(ti,'o:','linewidth',2,'MarkerFaceColor','m','MarkerEdgeColor','m');
                grid
                Cmin=min(min(dtt));
                Cmax=max(max(dtt));
                if Cmax>Cmin
                    h2=get(h1,'Parent');
                    set(h2,'YLim',[Cmin Cmax]);
                end
                title('Travel times differences','FontWeight','Bold','FontSize',12);
                xlabel('Index of travel path','FontWeight','Bold','FontSize',12);
                ylabel('\Deltat (ms)','FontWeight','Bold','FontSize',12);
                if Ntime>1
                    aviobj = VideoWriter('dtt.avi'); %#ok<TNMLP>
                    aviobj.Quality=50;
                    open(aviobj);
                end
                for i=1:Ntime
                    ti=dtt(:,i);
                    set(h1,'YData',ti);
                    if Ntime>1
                        frame = getframe(gcf);
                        writeVideo(aviobj,frame);
                    end
                end
                if Ntime>1
                    close(aviobj);
                end
            else
                warndlg('No data available!','Fig. info');
            end
            
        case 10 % s components and angles
            if exist('meanFields','var') & isfield(meanFields,'xy') & ~isempty(meanFields(1).xy)
                xy=meanFields(1).xy;
                for i=1:S % loop over all sources
                    for j=1:R % loop over all receivers
                        b=xy(S+j,1)-xy(i,1); % difference of x between the j-th receiver and i-th source
                        a=xy(S+j,2)-xy(i,2); % difference of y between the j-th receiver and i-th source
                        length_r=sqrt(a*a+b*b);
                        Sx=b/length_r; % x coordinate of the unit vector s - direction of the group velocity U
                        Sy=a/length_r; % y coordinate of the unit vector s
                        if Sx>=0 & Sy>=0
                            angle=rad2deg(asin(Sy));
                        elseif Sx<0 & Sy>=0
                            angle=rad2deg(pi-asin(Sy));
                        elseif Sx<0 & Sy<0
                            angle=rad2deg(-pi-asin(Sy));
                        elseif Sx>=0 & Sy<0
                            angle=rad2deg(asin(Sy));
                        end
                        theta(R*(i-1)+j,:)=[Sx Sy angle];
                    end
                end
                [Coeffs, pos]=sort(theta(:,1));
                figure;
                subplot(211)
                plot(Coeffs,'*-b','linewidth',2,'markersize',9);
                set(gca,'Ylim',[-1.2,1.2],'Ytick',[-1.2:0.2:1.2])
                grid
                title('s_x and s_y','FontWeight','Bold','FontSize',12);
                set(gca,'XTick',[1:size(theta,1)],'XTickLabel',pos);
                ylabel('s_{ix}','FontWeight','Bold','FontSize',12);
                [Coeffs, pos]=sort(theta(:,2));
                subplot(212)
                plot(Coeffs,'.-r','linewidth',2,'markersize',24);
                set(gca,'Ylim',[-1.2,1.2],'Ytick',[-1.2:0.2:1.2])
                set(gca,'XTick',[1:size(theta,1)],'XTickLabel',pos);
                grid
                xlabel('Index of travel path','FontWeight','Bold','FontSize',12);
                ylabel('s_{iy}','FontWeight','Bold','FontSize',12);
                
                [Coeffs, pos]=sort(theta(:,3));
                figure;
                plot(Coeffs,'*-b','linewidth',2,'markersize',9);
                grid
                title('Angles of rays','FontWeight','Bold','FontSize',12);
                xlabel('# of ray','FontWeight','Bold','FontSize',12);
                ylabel('Angles (deg)','FontWeight','Bold','FontSize',12);
                set(gca,'XTick',[1:size(theta,1)],'XTickLabel',pos);
                Ymin=round(min(theta(:,3)))-10;
                Ymax=round(max(theta(:,3)))+10;
                set(gca,'Ylim',[Ymin,Ymax],'Ytick',[Ymin:10:Ymax])
            else
                warndlg('No data available!','Fig. info');
            end
        case 11 % G_1 spectrum
            if exist('SingValG','var') & ~isempty(SingValG);
                figure;
                plot(SingValG,'linewidth',2);
                tex1=sprintf('G spectrum. S_{min}/S_{max} = %g',SingValG(end)/SingValG(1));
                title(tex1,'Fontsize',12,'FontWeigh','bold');
                xlabel('Index of singular value','Fontsize',12,'FontWeigh','bold');
                ylabel('Singular value','Fontsize',12,'FontWeigh','bold');
            else
                warndlg('No data available!','Fig. info');
            end
        case 12 % Constraints spectrum
            if exist('SingValC','var') & ~isempty(SingValC)
                
                if isfield(SingValC,'T') && ~isempty(SingValC.T)
                    figure;
                    plot(SingValC.T,'linewidth',2);
                    tex1=sprintf('C spectrum for T. S_{min}/S_{max} = %g',SingValC.T(end)/SingValC.T(1));
                    title(tex1,'Fontsize',12,'FontWeigh','bold');
                    xlabel('Index of singular value','Fontsize',12,'FontWeigh','bold');
                    ylabel('Singular value','Fontsize',12,'FontWeigh','bold');
                elseif isempty(SingValC.T)
                    warndlg('No T data available!','Fig. info');
                end
                if isfield(SingValC,'V') && ~isempty(SingValC.V)
                    figure;
                    plot(SingValC.V,'linewidth',2);
                    tex1=sprintf('C spectrum for V. S_{min}/S_{max} = %g',SingValC.V(end)/SingValC.V(1));
                    title(tex1,'Fontsize',12,'FontWeigh','bold');
                    xlabel('Index of singular value','Fontsize',12,'FontWeigh','bold');
                    ylabel('Singular value','Fontsize',12,'FontWeigh','bold');
                elseif isempty(SingValC.V)
                    warndlg('No V data available!','Fig. info');
                end
                if ~isfield(SingValC,'T') && ~isfield(SingValC,'V')
                    figure;
                    plot(SingValC,'linewidth',2);
                    tex1=sprintf('C spectrum. S_{min}/S_{max} = %g',SingValC(end)/SingValC(1));
                    title(tex1,'Fontsize',12,'FontWeigh','bold');
                    xlabel('Index of singular value','Fontsize',12,'FontWeigh','bold');
                    ylabel('Singular value','Fontsize',12,'FontWeigh','bold');
                end
            else
                warndlg('No data available!','Fig. info');
            end
        case 13 % dT expected NMSE & STD
            if exist('fluct','var') && isfield(fluct,'dTExpNMSE')
                uiwait(msgbox('NMSE of the fluctuations will be displayed now','Fig. info'));
                VX=fluct.dTExpNMSE;
                if interp>0
                    VX=interpn(VX,interp);
                end
                [M,grids,Ntime]=size(VX);
                v=VX(:,:,1);
                fig=figure;
                set(fig,'DoubleBuffer','on');
                h1=imagesc([xv(1),xv(end)],[yv(1),yv(end)],v');
                colormap(cMapName);
                axis equal;
                set(gca,'Xlim',[xv(1),xv(end)],'Ylim',[yv(1),yv(end)],'YDir','normal','Clim',[0 1]);
                title('NMSE: T field','FontWeight','Bold','FontSize',12);
                xlabel('x (m)','FontWeight','Bold','FontSize',12);
                ylabel('y (m)','FontWeight','Bold','FontSize',12);
                hc=colorbar;
                set(hc,'fontweight','bold');
                if Ntime>1
                    aviobj = VideoWriter('T_NMSE.avi'); %#ok<TNMLP>
                    aviobj.Quality=50;
                    open(aviobj);
                end
                for i=1:Ntime
                    v=VX(:,:,i);
                    set(h1,'CData',v');
                    if Ntime>1
                        frame = getframe(gcf);
                        writeVideo(aviobj,frame);
                    end
                end
                if Ntime>1
                    close(aviobj);
                end
            else
                warndlg('No NMSE data available!','Fig. info');
            end
            if exist('fluct','var') && isfield(fluct,'dTExpSTD')
                uiwait(msgbox('STD of full fields will be displayed now','Fig. info'));
                VX=fluct.dTExpSTD;
                Ntime=size(VX,3);
                if exist('meanFields','var') && isfield(meanFields,'std_dT')
                    for i=1:Ntime
                        VX(:,:,i)=sqrt(VX(:,:,i).^2+meanFields(i).std_dT^2);
                    end
                end
                if interp>0
                    VX=interpn(VX,interp);
                end
                [M,grids,Ntime]=size(VX);
                v=VX(:,:,1);
                fig=figure;
                set(fig,'DoubleBuffer','on');
                h1=imagesc([xv(1),xv(end)],[yv(1),yv(end)],v');
                colormap(cMapName);
                axis equal;
                set(gca,'Xlim',[xv(1),xv(end)],'Ylim',[yv(1),yv(end)],'YDir','normal');
                title('STD: T field','FontWeight','Bold','FontSize',12);
                xlabel('x (m)','FontWeight','Bold','FontSize',12);
                ylabel('y (m)','FontWeight','Bold','FontSize',12);
                Cmin=min(min(min(VX)));
                Cmax=max(max(max(VX)));
                if Cmax>Cmin
                    h2=get(h1,'Parent');
                    set(h2,'CLim',[Cmin Cmax]);
                end
                hc=colorbar;
                set(hc,'fontweight','bold');
                ht=get(hc,'Title');
                set(ht,'String','C^o','fontweight','bold','fontsize',12);
                if Ntime>1
                    aviobj = VideoWriter('T_STD.avi'); %#ok<TNMLP>
                    aviobj.Quality=50;
                    open(aviobj);
                end
                for i=1:Ntime
                    v=VX(:,:,i);
                    set(h1,'CData',v');
                    if Ntime>1
                        frame = getframe(gcf);
                        writeVideo(aviobj,frame);
                    end
                end
                if Ntime>1
                    close(aviobj);
                end
            else
                warndlg('No STD data available!','Fig. info');
            end
            
        case 14 % vx NMSE
            if exist('fluct','var') && isfield(fluct,'vxExpNMSE')
                uiwait(msgbox('NMSE of the fluctuations will be displayed now','Fig. info'));
                VX=fluct.vxExpNMSE;
                if interp>0
                    VX=interpn(VX,interp);
                end
                [M,grids,Ntime]=size(VX);
                v=VX(:,:,1);
                fig=figure;
                set(fig,'DoubleBuffer','on');
                h1=imagesc([xv(1),xv(end)],[yv(1),yv(end)],v');
                colormap(cMapName);
                axis equal;
                set(gca,'Xlim',[xv(1),xv(end)],'Ylim',[yv(1),yv(end)],'YDir','normal','Clim',[0 1]);
                title('NMSE: v_x field','FontWeight','Bold','FontSize',12);
                xlabel('x (m)','FontWeight','Bold','FontSize',12);
                ylabel('y (m)','FontWeight','Bold','FontSize',12);
                hc=colorbar;
                set(hc,'fontweight','bold');
                if Ntime>1
                    aviobj = VideoWriter('vx_NMSE.avi'); %#ok<TNMLP>
                    aviobj.Quality=50;
                    open(aviobj);
                end
                for i=1:Ntime
                    v=VX(:,:,i);
                    set(h1,'CData',v');
                    if Ntime>1
                        frame = getframe(gcf);
                        writeVideo(aviobj,frame);
                    end
                end
                if Ntime>1
                    close(aviobj);
                end
            else
                warndlg('No NMSE data available!','Fig. info');
            end
            if exist('fluct','var') && isfield(fluct,'vxExpSTD')
                uiwait(msgbox('STD of full fields will be displayed now','Fig. info'));
                VX=fluct.vxExpSTD;
                Ntime=size(VX,3);
                if exist('meanFields','var') && isfield(meanFields,'std_dvx')
                    for i=1:Ntime
                        VX(:,:,i)=sqrt(VX(:,:,i).^2+meanFields(i).std_dvx^2);
                    end
                end
                if interp>0
                    VX=interpn(VX,interp);
                end
                [M,grids,Ntime]=size(VX);
                v=VX(:,:,1);
                fig=figure;
                set(fig,'DoubleBuffer','on');
                h1=imagesc([xv(1),xv(end)],[yv(1),yv(end)],v');
                colormap(cMapName);
                axis equal;
                set(gca,'Xlim',[xv(1),xv(end)],'Ylim',[yv(1),yv(end)],'YDir','normal');
                title('STD: v_x field','FontWeight','Bold','FontSize',12);
                xlabel('x (m)','FontWeight','Bold','FontSize',12);
                ylabel('y (m)','FontWeight','Bold','FontSize',12);
                Cmin=min(min(min(VX)));
                Cmax=max(max(max(VX)));
                if Cmax>Cmin
                    h2=get(h1,'Parent');
                    set(h2,'CLim',[Cmin Cmax]);
                end
                hc=colorbar;
                set(hc,'fontweight','bold');
                ht=get(hc,'Title');
                set(ht,'String','m/s','fontweight','bold','fontsize',12);
                if Ntime>1
                    aviobj = VideoWriter('vx_STD.avi'); %#ok<TNMLP>
                    aviobj.Quality=50;
                    open(aviobj);
                end
                for i=1:Ntime
                    v=VX(:,:,i);
                    set(h1,'CData',v');
                    if Ntime>1
                        frame = getframe(gcf);
                        writeVideo(aviobj,frame);
                    end
                end
                if Ntime>1
                    close(aviobj);
                end
            else
                warndlg('No STD data available!','Fig. info');
            end
            
        case 15 % vy NMSE
            if exist('fluct','var') && isfield(fluct,'vyExpNMSE')
                uiwait(msgbox('NMSE of the fluctuations will be displayed now','Fig. info'));
                VX=fluct.vyExpNMSE;
                if interp>0
                    VX=interpn(VX,interp);
                end
                [M,grids,Ntime]=size(VX);
                v=VX(:,:,1);
                fig=figure;
                set(fig,'DoubleBuffer','on');
                h1=imagesc([xv(1),xv(end)],[yv(1),yv(end)],v');
                colormap(cMapName);
                axis equal;
                set(gca,'Xlim',[xv(1),xv(end)],'Ylim',[yv(1),yv(end)],'YDir','normal','Clim',[0 1]);
                title('NMSE: v_y field','FontWeight','Bold','FontSize',12);
                xlabel('x (m)','FontWeight','Bold','FontSize',12);
                ylabel('y (m)','FontWeight','Bold','FontSize',12);
                hc=colorbar;
                set(hc,'fontweight','bold');
                if Ntime>1
                    aviobj = VideoWriter('vy_NMSE.avi'); %#ok<TNMLP>
                    aviobj.Quality=50;
                    open(aviobj);
                end
                for i=1:Ntime
                    v=VX(:,:,i);
                    set(h1,'CData',v');
                    if Ntime>1
                        frame = getframe(gcf);
                        writeVideo(aviobj,frame);
                    end
                end
                if Ntime>1
                    close(aviobj);
                end
            else
                warndlg('No NMSE data available!','Fig. info');
            end
            if exist('fluct','var') && isfield(fluct,'vyExpSTD')
                uiwait(msgbox('STD of full fields will be displayed now','Fig. info'));
                VX=fluct.vyExpSTD;
                Ntime=size(VX,3);
                if exist('meanFields','var') && isfield(meanFields,'std_dvy')
                    for i=1:Ntime
                        VX(:,:,i)=sqrt(VX(:,:,i).^2+meanFields(i).std_dvy^2);
                    end
                end
                if interp>0
                    VX=interpn(VX,interp);
                end
                [M,grids,Ntime]=size(VX);
                v=VX(:,:,1);
                fig=figure;
                set(fig,'DoubleBuffer','on');
                h1=imagesc([xv(1),xv(end)],[yv(1),yv(end)],v');
                colormap(cMapName);
                axis equal;
                set(gca,'Xlim',[xv(1),xv(end)],'Ylim',[yv(1),yv(end)],'YDir','normal');
                title('STD: v_y field','FontWeight','Bold','FontSize',12);
                xlabel('x (m)','FontWeight','Bold','FontSize',12);
                ylabel('y (m)','FontWeight','Bold','FontSize',12);
                Cmin=min(min(min(VX)));
                Cmax=max(max(max(VX)));
                if Cmax>Cmin
                    h2=get(h1,'Parent');
                    set(h2,'CLim',[Cmin Cmax]);
                end
                hc=colorbar;
                set(hc,'fontweight','bold');
                ht=get(hc,'Title');
                set(ht,'String','m/s','fontweight','bold','fontsize',12);
                set(hc,'fontweight','bold');
                if Ntime>1
                    aviobj = VideoWriter('vy_STD.avi'); %#ok<TNMLP>
                    aviobj.Quality=50;
                    open(aviobj);
                end
                for i=1:Ntime
                    v=VX(:,:,i);
                    set(h1,'CData',v');
                    if Ntime>1
                        frame = getframe(gcf);
                        writeVideo(aviobj,frame);
                    end
                end
                if Ntime>1
                    close(aviobj);
                end
            else
                warndlg('No STD data available!','Fig. info');
            end
        case 16 % V NMSE
            if exist('fluct','var') && isfield(fluct,'vx') && isfield(fluct,'vy')
                uiwait(msgbox('STD of full fields will be displayed now','Fig. info'));
                V1=fluct.vx;
                V2=fluct.vy;
                [M,grids,Ntime]=size(V1);
                if exist('meanFields','var')&& isfield(meanFields,'vx') && isfield(meanFields,'vy')
                    for i=1:Ntime
                        V1(:,:,i)=V1(:,:,i)+meanFields(i).vx;
                        V2(:,:,i)=V2(:,:,i)+meanFields(i).vy;
                    end
                end
                %                 if interp>0
                %                     V1=interpn(V1,interp);
                %                     V2=interpn(V2,interp);
                %                     Ntime=size(V1,3);
                %                 end
                V=sqrt(V1.^2+V2.^2);
                dVX2=fluct.vxExpSTD.^2;
                dVY2=fluct.vyExpSTD.^2;
                if exist('meanFields','var') && isfield(meanFields,'std_dvy')
                    for i=1:Ntime
                        dVX2(:,:,i)=dVX2(:,:,i)+meanFields(i).std_dvx^2;
                        dVY2(:,:,i)=dVY2(:,:,i)+meanFields(i).std_dvy^2;
                    end
                end
                % std of the V, denoted by VX
                VX=sqrt(V1.^2.*dVX2+V2.^2.*dVY2+2*dVX2.*dVY2+0.25*(dVX2.^2+dVY2.^2))./(V+eps);
                if interp>0
                    VX=interpn(VX,interp);
                end
                [M,grids,Ntime]=size(VX);
                v=VX(:,:,1);
                fig=figure;
                set(fig,'DoubleBuffer','on');
                h1=imagesc([xv(1),xv(end)],[yv(1),yv(end)],v');
                colormap(cMapName);
                axis equal;
                set(gca,'Xlim',[xv(1),xv(end)],'Ylim',[yv(1),yv(end)],'YDir','normal');
                title('STD: V field','FontWeight','Bold','FontSize',12);
                xlabel('x (m)','FontWeight','Bold','FontSize',12);
                ylabel('y (m)','FontWeight','Bold','FontSize',12);
                Cmin=min(min(min(VX)));
                Cmax=max(max(max(VX)));
                if Cmax>Cmin
                    h2=get(h1,'Parent');
                    set(h2,'CLim',[Cmin Cmax]);
                end
                hc=colorbar;
                set(hc,'fontweight','bold');
                ht=get(hc,'Title');
                set(ht,'String','m/s','fontweight','bold','fontsize',12);
                set(hc,'fontweight','bold');
                if Ntime>1
                    aviobj = VideoWriter('V_STD.avi'); %#ok<TNMLP>
                    aviobj.Quality=50;
                    open(aviobj);
                end
                for i=1:Ntime
                    v=VX(:,:,i);
                    set(h1,'CData',v');
                    if Ntime>1
                        frame = getframe(gcf);
                        writeVideo(aviobj,frame);
                    end
                end
                if Ntime>1
                    close(aviobj);
                end
            else
                warndlg('No STD data available!','Fig. info');
            end
    end % switch
end % while
