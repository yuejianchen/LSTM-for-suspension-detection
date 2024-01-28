% clear;clc;close all;
set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 11);
path(path,'D:\ShareCache\陈岳剑(21155)\Assistant Prof R&D\Deep LSTM\MST-Deep LSTM\Codes, Data, & Results\Wheel-Rail Dynamic Model and Data');
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

global XTrain YTrain XValid YValid Nini;
XTrain = [];YTrain = [];XValid = [];YValid = []; Nini= [];

% prepare training and validation data
for i = 1:50
    XTemp = (downsample(IFFTfilter(xp{i},fs,fc),ratio)-MEAN)/STD;
    XTrain{i} = XTemp(1:round(end/2)-1);
    YTrain{i} = XTemp(2:round(end/2));
    XTemp = (downsample(IFFTfilter(xp{50+i},fs,fc),ratio)-MEAN)/STD;
    XValid{i} = XTemp(1:round(end/2)-1);
    YValid{i} = XTemp(2:round(end/2));
end

Nini = 20;

%% Enumrate selection
for i = 1:2^10 
    i
    msevalid(i) = ObjFun(uint16(i-1));
end
[msemin,x] = min(msevalid);

figure;hold on;grid on;
plot(msevalid(1:end/2),'-*');%Learning Rate = 0.005
plot(msevalid(end/2+1:end),'-*');%Learning Rate = 0.01

%% Find the optimal neural network given number of layers
temp = 1:256;%initiate sequence of neural networks indexes under a specific number of layers
temp = dec2bin(temp-1);% convert to binary
temp2 = [];
for i = 1:256 % convert uint8 to uint10 by incoporating the number of layers
    temp2(i) = bin2dec([temp(i,1:3) '00' temp(i,4:end)])+1;%'00' denotes layer = 2
end
[~,x] = min(msevalid(temp2));
x = temp2(x);

%% Output final neural network
ind = dec2bin(x-1);
if length(ind)<10
   for i = 1:(10-length(ind))
        ind = ['0' ind ];
   end
end

indLR = uint8(bin2dec(ind(1))+1);
indL2 = uint8(bin2dec(ind(2:3))+1);
indHl = uint8(bin2dec(ind(4:5))+1);
indHi = uint8(bin2dec(ind(6:10))+1);

LearnRate = [0.005 0.001];
L2Reg = [1e-1 1e-2 1e-3 1e-4];
numHiddenLayers = [2 3 4 5];
numHiddenUnits = [10:10:100 150:50:1200];

global XTrain YTrain XValid YValid Nini;
% Stacked LSTM modeling
numFeatures = 1;
numResponses = 1;
switch numHiddenLayers(indHl)
    case 2
        layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits(indHi))
        lstmLayer(ceil(numHiddenUnits(indHi)/2))
        fullyConnectedLayer(numResponses)
        regressionLayer];
    case 3
        layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits(indHi))
        lstmLayer(ceil(numHiddenUnits(indHi)/2))
        lstmLayer(ceil(numHiddenUnits(indHi)/4))
        fullyConnectedLayer(numResponses)
        regressionLayer];
    case 4
        layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits(indHi))
        lstmLayer(ceil(numHiddenUnits(indHi)/2))
        lstmLayer(ceil(numHiddenUnits(indHi)/4))
        lstmLayer(ceil(numHiddenUnits(indHi)/8))
        fullyConnectedLayer(numResponses)
        regressionLayer];
    case 5
        layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits(indHi))
        lstmLayer(ceil(numHiddenUnits(indHi)/2))
        lstmLayer(ceil(numHiddenUnits(indHi)/4))
        lstmLayer(ceil(numHiddenUnits(indHi)/8))
        lstmLayer(ceil(numHiddenUnits(indHi)/16))
        fullyConnectedLayer(numResponses)
        regressionLayer];
end

options = trainingOptions('adam', ...
    'MaxEpochs',500, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',LearnRate(indLR), ...%0.005
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'L2Regularization',L2Reg(indL2), ...
    'ExecutionEnvironment','gpu',...
    'Verbose',0);
    % 'ExecutionEnvironment','gpu',...
    % 'Plots','training-progress',...
    % 'ValidationFrequency',10,...
    % 'ValidationData',{XValid,YValid});

rng(1);%Ensure the training process repeatable
net = trainNetwork(XTrain,YTrain,layers,options);

% Training MSE
YPred = predict(net,XTrain);
for ii = 1:length(YPred)
    res(ii,:) = YTrain{ii}(Nini:end)-YPred{ii}(Nini:end);
    MSE_train(ii)=mse(res(ii,:));
end
(mean(MSE_train))^.5

% Validation MSE
YPred = predict(net,XValid);
for ii = 1:length(YPred)
    res(ii,:) = YValid{ii}(Nini:end)-YPred{ii}(Nini:end);
    MSE_lstm(ii)=mse(res(ii,:));
end
(mean(MSE_lstm))^.5

timeend = num2str(datestr(now,'dd-mmm-yyyy HH:MM:SS'))
% save 20201108StackedLSTM.mat;
save 20210106StackedLSTM(Layers=2).mat;%Nl=2
%% Fault Detection
% TBA