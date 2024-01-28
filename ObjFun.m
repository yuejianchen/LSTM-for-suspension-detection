function msevalid = ObjFun(ind)
ind = dec2bin(ind);

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
    'MaxEpochs',5, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',LearnRate(indLR), ...%0.005
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'L2Regularization',L2Reg(indL2), ...
    'ExecutionEnvironment','cpu',...
    'Verbose',0);
 
    % 'Plots','training-progress',...
    % 'MiniBatchSize',128,...
    % 'ValidationFrequency',10,...
    % 'ValidationData',{XValid,YValid},...        
    
rng(1);%Ensure the training process repeatable
net = trainNetwork(XTrain,YTrain,layers,options);

% Validation MSE
YPred = predict(net,XValid);
for ii = 1:length(YPred)
    res(ii,:) = YValid{ii}(Nini:end)-YPred{ii}(Nini:end);
    MSE_lstm(ii)=mse(res(ii,:));
end
msevalid = mean(MSE_lstm);
end