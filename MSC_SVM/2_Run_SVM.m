% This code runs across all subjects, imports the MSC matrices to create the feature matrix, train and test a quadratic polynomial SVM. 

% Returns the accuracy, precision, recall, F1 score, train and test times. 

%Last modified: 17/03/2025 by Costanza Armanini
%%
%[text] ## Initial settings
numSubjects = 2; % Number of subjects
numMovements = 49; % Number of movements
numReps = 6; % Number of repetitions
numSensors = 12; % Number of sensors

accuracy_results = zeros(numSubjects, 4);  % Subject Label, Accuracy, Train Time, Test Time
%%
%[text] ## Create feature matrix, train and test the SVM and save results
for subject = 1:numSubjects %for cycle across al subjects %[output:group:81e85f16]
    fprintf('Processing Subject %d...\n', subject); %[output:37b94d6b]
    
    % Initialize the feature matrix and vectors for labels and repetitions
    
    numFeaturesPerPair = 1; % Assuming 1 feature: mean of MSC
    numPairs = numSensors * (numSensors - 1);
    featureMatrix = zeros(numMovements * numReps, numPairs * numFeaturesPerPair);
    labels = zeros(numMovements * numReps, 1); % Labels vector
    repetitions = zeros(numMovements * numReps, 1); % Repetition number vector
    
    
    % Load and process each file
    rowIdx = 1;
    for mov = 1:numMovements %import the MSC matrix for each movement
        filename = fullfile(sprintf('Subject%d', subject), sprintf('MSC_S%d_mov%d.mat', subject, mov));
        data = load(filename); %[output:6845d233]
        MSC_allReps = data.MSC_allReps;
    
        for rep = 1:numReps
                MSC_matrix = MSC_allReps{rep};
                featureRow = [];
        
                for s1 = 1:numSensors
                    for s2 = 1:numSensors
        
                        if s1 ~= s2 % Exclude self-pairings
                            mscValues = squeeze(MSC_matrix(s1, s2, :));
        
                            % Calculate mean 
                            meanVal = mean(mscValues);
        
                            % Collect all features for this sensor pair
                            featureRow = [featureRow, meanVal];
                        end
                    end
                end
                
                featureMatrix(rowIdx, :) = featureRow;
                labels(rowIdx) = mov;  % Movement label
                repetitions(rowIdx) = rep;  % Repetition number
                rowIdx = rowIdx + 1;
        end
    end

    % Specify repetitions to use for training and testing
    trainReps = [1, 3, 4, 6];
    testReps = [2, 5];
    
    % Find indices for training and testing based on specified repetitions
    trainIdx = ismember(repetitions, trainReps);
    testIdx = ismember(repetitions, testReps);
    
    % Extract training features and labels
    X_train = featureMatrix(trainIdx, :);
    y_train = labels(trainIdx);
    
    % Extract testing features and labels
    X_test = featureMatrix(testIdx, :);
    y_test = labels(testIdx);
    
    % Normalize features
    X_train_norm = (X_train - mean(X_train)) ./ std(X_train);
    X_test_norm = (X_test - mean(X_train)) ./ std(X_train);  % Use training mean and std for normalization
   
    
    % SVM template
    bestOrder =2; %Polynomial order
    bestC = 10; %Regularization parameter

    template_final = templateSVM(...
        'KernelFunction','polynomial',...           %polynomial Kernel
        'PolynomialOrder',bestOrder,...             
        'KernelScale','auto',...
        'BoxConstraint',bestC,...
        'Standardize',true);

    % Train the SVM classifier
    tic;  % Start timer for training
    SVMModel_final = fitcecoc(X_train_norm, y_train, 'Learners', template_final, 'Coding','onevsall', 'ObservationsIn', 'rows');
    training_time = toc;  % End timer for training
    
    % Test the SVM classifier
    predictedLabels_Train = predict(SVMModel_final, X_train_norm);
    predictedLabels = predict(SVMModel_final, X_test_norm);
    
    tic;  % Start timer for testing
    [~, scores] = predict(SVMModel_final, X_test_norm); % Get predicted scores for AUC
    testing_time = toc;  % End timer for testing

    % Calculate overall accuracy
    accuracy_train = sum(predictedLabels_Train == y_train) / length(y_train);
    accuracy = sum(predictedLabels == y_test) / length(y_test);
    
    %% Store results (subject, accuracy, training time, testing time)
    accuracy_results(subject, :) = [subject, accuracy * 100, training_time, testing_time];
    
    fprintf('Subject %d - Test Accuracy: %.2f%%\n', subject, accuracy * 100);
    
    % Calculate accuracy for each movement
        for mov = 1:numMovements
            movementIdx = (y_test == mov);
            movementAccuracy(subject, mov) = sum(predictedLabels(movementIdx) == y_test(movementIdx)) / sum(movementIdx);
        end
    
        
    % Calculate precision, recall, and F1-score
    [confMat,order] = confusionmat(y_test, predictedLabels);
    
    % Calculate Precision, Recall, and F1-Score for each class
    numClasses = length(order);  % Number of classes
    precision = zeros(numClasses, 1);
    recall = zeros(numClasses, 1);
    F1 = zeros(numClasses, 1);
    TPR = zeros(numClasses, 1);
    FPR = zeros(numClasses, 1);
    
    for i = 1:numClasses
        TP = confMat(i, i);
        FP = sum(confMat(:, i)) - TP;
        FN = sum(confMat(i, :)) - TP;
        TN = sum(confMat(:)) - TP - FP - FN;
    
        precision(i) = confMat(i,i)/sum(confMat(:,i))*100;
        recall(i) = confMat(i,i)/sum(confMat(i,:))*100;
        F1(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
        TPR(i) = TP / (TP + FN);
        FPR(i) = FP / (FP + TN);
    end
    
    % Handle NaN values if any division by zero occurs
    precision(isnan(precision)) = 0;
    recall(isnan(recall)) = 0;
    F1(isnan(F1)) = 0;
    
    % Save results, true labels, and predicted scores for AUC calculation
        filename = fullfile(sprintf('Results_S%d.mat', subject));
        save(filename, 'accuracy',  'movementAccuracy', 'confMat', 'precision', 'recall', 'F1', 'y_test', 'scores');

end %[output:group:81e85f16]
%%
%[text] ## Save Accuracy, Train and Test Times
accuracy_table = array2table(accuracy_results, 'VariableNames', {'Subject', 'Accuracy', 'TrainingTime', 'TestingTime'});
disp(accuracy_table);
writetable(accuracy_table, 'MSC_Accuracy_and_Times.csv');

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":31.2}
%---
%[output:37b94d6b]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 1...\n","truncated":false}}
%---
%[output:6845d233]
%   data: {"dataType":"error","outputData":{"errorType":"runtime","text":"Error using <a href=\"matlab:matlab.lang.internal.introspective.errorDocCallback('load')\" style=\"font-weight:bold\">load<\/a>\nSubject1\\MSC_S1_mov1.mat is not found in the current folder or on the MATLAB path, but exists in:\n    C:\\Users\\ca3072\\Desktop\\NYU Project 5\\Muscular Coherence Paper\\Matlab Code\\Final_Code_Total_Time\\New_All_Gesturres\\MSC_SVM\\2_SVM_Classification\n    C:\\Users\\ca3072\\Desktop\\NYU Project 5\\Muscular Coherence Paper\\Matlab Code\\Final_Code_Total_Time\\New_All_Gesturres\\MSC_SVM\\MSC_Networks_Plots\n    C:\\Users\\ca3072\\Desktop\\NYU Project 5\\Muscular Coherence Paper\\Matlab Code\\Final_Code_Total_Time\\New_All_Gesturres\\MSC_SVM\\1_MSC_Calculation\n\nChange the MATLAB current folder or add its folder to the MATLAB path."}}
%---
