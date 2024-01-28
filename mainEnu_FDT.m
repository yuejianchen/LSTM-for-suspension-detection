clear;clc;close all;
path(path,'G:\My Drive\2. Posdoc Research and Exp Design\Topic 1 Anormaly Detection with advanced ML\Wheel-Rail Dynamic Model and Data');
% load('20201108StackedLSTM.mat');
load('20210106StackedLSTM(Layers=2).mat');
xf = xp;
load('Ksz1-10redu_NoRail_dur100_seed301-500.mat');
xf = [xf xp];
load('Ksz1-20redu_NoRail_dur100_seed501-700.mat');
xf = [xf xp];
load('Ksz1-30redu_NoRail_dur100_seed701-900.mat');
xf = [xf xp];
% load('Csz1-10redu_NoRail_dur100_seed901-1100.mat');
% xf = [xf xp];
% load('Csz1-20redu_NoRail_dur100_seed1101-1300.mat');
% xf = [xf xp];
% load('Csz1-30redu_NoRail_dur100_seed1301-1500.mat');
% xf = [xf xp];
clear xp;
Nini = 101;
%% Fault Detection
Ntotal = 800;
for i = 1:Ntotal
    XTemp = (downsample(IFFTfilter(xf{i+101}(9,:),fs,fc),ratio)-MEAN)/STD;
    XFault = XTemp(1:round(end/2)-1);
    YFault = XTemp(2:round(end/2));
    
    % Predication
    YFaultPred = predict(net,XFault);
    YFault = YFault(Nini:end);
    YFaultPred = YFaultPred(Nini:end);  
    EFault = YFault-YFaultPred;
   
    MSE_lstmAE(i)=mse(EFault);
    MAE_lstmAE(i)=mean(abs(EFault));
    rmse_lstmAE(i)=sqrt(sum((EFault).^2)/length(EFault));
    
    mdl = fitlm(YFault,YFaultPred);
    Rsquared(i) = mdl.Rsquared.Ordinary;
    
    % if i == 1 || i == 201
    %     figure;hold on;
    %     plot(YFault,'-.');
    %     plot(YFaultPred);
    %     plot(EFault,'k');
    %     legend('Raw','Reconst','Res');
    % 
    %     figure;
    %     plot(YFault,YFaultPred,'*');
    %     xlabel('YFault');
    %     %set(gcf,'position',[400,320,wid*ratiow,wid*1.1]);
    %     %hold on;
    %     plot(mdl.Fitted,YFault,'linewidth',2);
    %     ylabel('YFaultPred');
    %     legend('raw data','linear fit','location','nw');
    %     title(['R^2=' num2str(mdl.Rsquared.Ordinary,3)]); 
    % end
end

figure;
subplot(311);title AE;
plot(MSE_lstmAE);ylabel MSE;
subplot(312);
plot(MAE_lstmAE);ylabel MAE;
subplot(313);
plot(Rsquared);ylabel R2AE;
xlabel('Num of Samples');

% ROC plots
clusterTargets = [ones(1,200),zeros(1,200)];
for ii = 1:3
    clusterOutputs = [Rsquared(1:200),Rsquared(((ii)*200+1):(ii+1)*200)];
    figure;
    plotroc(clusterTargets, clusterOutputs);
end
clear xf;
save 20210106StackedLSTM(Layers=2)FDT_Nini101.mat;