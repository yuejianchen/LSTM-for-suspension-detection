clear;clc;close all;
set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 11);
path(path,'G:\My Drive\2. Posdoc Research and Exp Design\Topic 1 Anormaly Detection with advanced ML\Wheel-Rail Dynamic Model and Data');
timestart = num2str(datestr(now,'dd-mmm-yyyy HH:MM:SS'))
%% Data loading
load('healthyNoRail_dur100_seed1-301_redu.mat');

fs = 200;fc = 5;ratio = fs/(2*fc);
X = downsample(IFFTfilter(xp{1},fs,fc),ratio);%9 - car body vibration
MEAN = mean(X);
STD = std(X);
X = (X-MEAN)/STD;
% fs2 = fs/ratio;
% dt = 1/fs2;
% t = dt:dt:((length(X)-1)/2)/fs2;

% T = 50;dt = 0.1;
% t = dt:dt:T-dt;
% figure;
% ax1=subplot(211);
% plot(t,(downsample(IFFTfilter(xp{1}(9,1:end/2),fs,fc),ratio)-MEAN)/STD);
% xlabel('Time (s)');ylabel('Acceleration (m/s^2)');
% ax2=subplot(212);
% fftspecturm((downsample(IFFTfilter(xp{1}(9,1:end/2),fs,fc),ratio)-MEAN)/STD,10,1);

XTrain = [];XValid = [];

% prepare training and validation data
for i = 1:50
    XTemp = (downsample(IFFTfilter(xp{i},fs,fc),ratio)-MEAN)/STD;
    XTrain{i} = XTemp(1:round(end/2));
    XTemp = (downsample(IFFTfilter(xp{50+i},fs,fc),ratio)-MEAN)/STD;
    XValid{i} = XTemp(1:round(end/2));
end

%% AR modeling
for na = 1:100
    na
    [~,~,BIC(na),~] = ARX_LS_Estimation(XTrain,0,na,0,'AR');
end
[BICmin,na] = min(BIC);
figure;plot(BIC);
[a,~,~,~] = ARX_LS_Estimation(XTrain,0,na,0,'AR');
e = ARX_eval(a,0,XValid,0,na,0,'AR');
msevalid = mse(e);

timeend = num2str(datestr(now,'dd-mmm-yyyy HH:MM:SS'))
save 20201112AR.mat;

%% Fault Detection
% load('healthyNoRail_dur100_seed1-301.mat');
xf = xp;
load('Ksz1-10redu_NoRail_dur100_seed301-500_redu.mat');
xf = [xf xp];
load('Ksz1-20redu_NoRail_dur100_seed501-700_redu.mat');
xf = [xf xp];
load('Ksz1-30redu_NoRail_dur100_seed701-900_redu.mat');
xf = [xf xp];
% load('Csz1-10redu_NoRail_dur100_seed901-1100.mat');
% xf = [xf xp];
% load('Csz1-20redu_NoRail_dur100_seed1101-1300.mat');
% xf = [xf xp];
% load('Csz1-30redu_NoRail_dur100_seed1301-1500.mat');
% xf = [xf xp];
clear xp;
Nini = 101;
%
Ntotal = 800;
for i = 1:Ntotal
    i
    XTemp = (downsample(IFFTfilter(xf{i+101},fs,fc),ratio)-MEAN)/STD;
    XFault = XTemp(1:round(end/2));
    
    % Predication
    EFault = ARX_eval(a,0,XFault',0,na,0,'AR');
    XFaultMdl = XFault' - EFault;
    
    EFault = EFault(Nini:end);
    XFaultMdl = XFaultMdl(Nini:end);
    XFault = XFault(Nini:end);
    
    MSE_AR(i)=mse(EFault);
    MAE_AR(i)=mean(abs(EFault));
    rmse_AR(i)=sqrt(sum((EFault).^2)/length(EFault));
    
    mdl = fitlm(XFault,XFaultMdl);
    Rsquared(i) = mdl.Rsquared.Ordinary;
    
    if i == 1 || i == 201
        figure;hold on;
        plot(XFault,'-.');
        plot(XFaultMdl);
        plot(EFault,'k');
        legend('Raw','Reconst','Res');

        figure;
        plot(XFault,XFaultMdl,'*');
        xlabel('XFault');
        %set(gcf,'position',[400,320,wid*ratiow,wid*1.1]);
        %hold on;
        plot(mdl.Fitted,XFault,'linewidth',2);
        ylabel('XFaultMdl');
        legend('raw data','linear fit','location','nw');
        title(['R^2=' num2str(mdl.Rsquared.Ordinary,3)]); 
    end
end

figure;
subplot(311);title AR;
plot(MSE_AR);ylabel MSE;
subplot(312);
plot(rmse_AR);ylabel RMSE;
subplot(313);
plot(Rsquared);ylabel R2;
xlabel('Num of Samples');

% ROC plots
clusterTargets = [ones(1,200),zeros(1,200)];
for ii = 1:3
    clusterOutputs = [Rsquared(1:200),Rsquared(((ii)*200+1):(ii+1)*200)];
    figure;
    plotroc(clusterTargets, clusterOutputs);
end
clear xf;
save 20210110AR-FDT_Nini101.mat;
%%
T = 50;dt = 0.1;
t = dt:dt:T-dt;
figure;
ax1=subplot(421);
plot(t,(downsample(IFFTfilter(xf{1}(1:end/2),fs,fc),ratio)-MEAN)/STD);
xlabel('Time (s)');ylabel('Acceleration (m/s^2)');
ax2=subplot(422);
fftspecturm((downsample(IFFTfilter(xf{1}(1:end/2),fs,fc),ratio)-MEAN)/STD,10,1);

ax3=subplot(423);
plot(t,(downsample(IFFTfilter(xf{302}(1:end/2),fs,fc),ratio)-MEAN)/STD);
xlabel('Time (s)');ylabel('Acceleration (m/s^2)');
ax4=subplot(424);
fftspecturm((downsample(IFFTfilter(xf{302}(1:end/2),fs,fc),ratio)-MEAN)/STD,10,1);

ax5=subplot(425);
plot(t,(downsample(IFFTfilter(xf{502}(1:end/2),fs,fc),ratio)-MEAN)/STD);
xlabel('Time (s)');ylabel('Acceleration (m/s^2)');
ax6=subplot(426);
fftspecturm((downsample(IFFTfilter(xf{502}(1:end/2),fs,fc),ratio)-MEAN)/STD,10,1);

ax7=subplot(427);
plot(t,(downsample(IFFTfilter(xf{702}(1:end/2),fs,fc),ratio)-MEAN)/STD);
xlabel('Time (s)');ylabel('Acceleration (m/s^2)');
ax8=subplot(428);
fftspecturm((downsample(IFFTfilter(xf{702}(1:end/2),fs,fc),ratio)-MEAN)/STD,10,1);

linkaxes([ax1 ax3 ax5 ax7],'xy');
linkaxes([ax2 ax4 ax6 ax8],'xy');