
% SEE https://www.mathworks.com/help/nnet/examples/create-simple-deep-learning-network-for-classification.html
 
clear all; close all;


%--------------------------

% load input data
fprintf(1,'Loading data... ');
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
        'nndatasets','DigitDataset');
digitData = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
fprintf(1,'done\n');   

%--------------------------
        
% how many classes
numClasses = size(digitData.countEachLabel,1);

% size of image (all the same)
[numRows, numCols, numChannels]=size(readimage(digitData,1));

%--------------------------

% show samples
perm = randperm(length(digitData.Files),20);
for i = 1:20
    subplot(4,5,i);
    imshow(digitData.Files{perm(i)});
end
pause(3); close;

%--------------------------

% split into training/testing - do not change!
trainFraction = .5;
testFraction = .5;
rng('default'); % For reproducibility
[trainDigitData,testDigitData ] = splitEachLabel(digitData, ...
				trainFraction, testFraction, 'randomize');
    
numTrain = length(trainDigitData.Labels);
numTest = length(testDigitData.Labels);

%--------------------------
%--------------------------
%--------------------------

% ALTER THIS PORTION OF CODE ONLY!

% name this version of the model (no spaces)
modelName = 'model12'; 

% define the CNN
Layers = [imageInputLayer([numRows numCols numChannels])
          convolution2dLayer(2, 32)
          reluLayer
          maxPooling2dLayer(2, 'Stride', 2)
          fullyConnectedLayer(512)
          reluLayer
          fullyConnectedLayer(numClasses)
          softmaxLayer
          classificationLayer()];

% total number of epochs to train
maxEpochs = 15;

%--------------------------
%--------------------------
%--------------------------

% get 1-hot vector targets
trainTargets = zeros(numClasses, numTrain);
for i=1:numTrain
    n=double(trainDigitData.Labels(i)); % increments category by 1
    trainTargets(n,i)=1;
end

%--------------------------

 % training options 
options = trainingOptions('sgdm','MaxEpochs', maxEpochs, ...
    'MiniBatchSize', 512, 'L2Regularization', 0.0001, ...
	'InitialLearnRate', 0.001, 'Momentum', 0.9, ...
    'Verbose', true, 'VerboseFrequency', 200); 

% train it 
tic;
convnet = trainNetwork(trainDigitData, Layers, options);
elapsedTime = toc;
fprintf(1,'Elapsed Time: %2.2f minutes\n', elapsedTime/60.0);

%--------------------------
    
% test score with best val network
YTest = classify(convnet, testDigitData);
TTest = testDigitData.Labels;
TestAcc = 100.0 * sum(YTest == TTest)/numel(TTest);
fprintf(1,'Test Accuracy: %2.2f%%\n', TestAcc);

%--------------------------
    
% save it
fname = sprintf('cnn_%s.mat', modelName);
save(fname,'convnet', 'TestAcc', 'elapsedTime', '-v7.3');


