numSubjects = 40;
accuracy_results = zeros(numSubjects, 4);  % Add columns for training and testing times
%%
for subject = 6:numSubjects %[output:group:020be53e]
    fprintf('Processing Subject %d...\n', subject);    %[output:9a73d74c]

    % Load sEMG Data
    Emg = readtable(['emg_normalized_subject', num2str(subject), '_All.csv']); %[output:40cfd4b0]
    Emg(1,:) = [];
    Emg(:,1) = [];
    Emg = Emg{:,:};

    Movements = readtable(['move_labels_subject', num2str(subject), '_All.csv']);
    Movements(1,:) = [];
    Movements(:,1) = [];
    Movements = Movements{:,:};

    Rep = readtable(['rep_labels_subject', num2str(subject), '_All.csv']);
    Rep(1,:) = [];
    Rep(:,1) = [];
    Rep = Rep{:,:}; 

    %% Define Parameters
    fs = 2000;
    window_length = 600; 
    overlap = 0.5 * window_length; 
    step_size = window_length - overlap;

    %% Prepare Data for CNN
    X = [];
    Y = [];
    Reps = [];  % Store repetition labels separately for training/testing split

    for start_idx = 1:step_size:(size(Emg, 1) - window_length + 1)
        end_idx = start_idx + window_length - 1;
        window_data = Emg(start_idx:end_idx, :);
        move_label = mode(Movements(start_idx:end_idx)); 
        rep_label = mode(Rep(start_idx:end_idx)); 

        X = cat(4, X, reshape(window_data, [window_length, 1, 12, 1]));
        Y = [Y; move_label];  
        Reps = [Reps; rep_label]; 
    end

    %% Split into Training and Testing Sets based on Repetitions
    train_idx = ismember(Reps, [1, 3, 4, 6]); % Train on repetitions 1, 3, 4, 6
    test_idx = ismember(Reps, [2, 5]);       % Test on repetitions 2, 5

    X_train = X(:,:,:,train_idx);
    y_train = categorical(Y(train_idx)); % Convert movement labels to categorical
    X_test = X(:,:,:,test_idx);
    y_test = categorical(Y(test_idx));

    %% Define CNN Architecture
    layers = [ 
        imageInputLayer([window_length, 1, 12])
        
        convolution2dLayer([5,1], 16, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        
        convolution2dLayer([5,1], 32, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        
        fullyConnectedLayer(49)  % Output layer size should match the number of movements
        softmaxLayer
        classificationLayer
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 10, ... 
        'MiniBatchSize', 64, ...
        'Verbose', false, ...
        'Plots', 'training-progress');

    %% Measure Training Time
    tic;  % Start timer for training
    cnn_model = trainNetwork(X_train, y_train, layers, options);
    training_time = toc;  % End timer for training

    %% Measure Testing Time
    tic;  % Start timer for testing
    y_pred = classify(cnn_model, X_test);
    testing_time = toc;  % End timer for testing

    accuracy = sum(y_pred == y_test) / length(y_test);
    fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

    %% Store results (subject, accuracy, training time, testing time)
    accuracy_results(subject, :) = [subject, accuracy * 100, training_time, testing_time];
end %[output:group:020be53e]

%%
%% Save Accuracy and Time Table
accuracy_table = array2table(accuracy_results, 'VariableNames', {'Subject', 'Accuracy', 'TrainingTime', 'TestingTime'});
disp(accuracy_table);
writetable(accuracy_table, 'CNN_Accuracy_and_Times.csv');


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":28.3}
%---
%[output:9a73d74c]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 6...\n","truncated":false}}
%---
%[output:40cfd4b0]
%   data: {"dataType":"error","outputData":{"errorType":"runtime","text":"Error using <a href=\"matlab:matlab.lang.internal.introspective.errorDocCallback('readtable', 'C:\\Program Files\\MATLAB\\R2024b\\toolbox\\matlab\\iofun\\readtable.m', 517)\" style=\"font-weight:bold\">readtable<\/a> (<a href=\"matlab: opentoline('C:\\Program Files\\MATLAB\\R2024b\\toolbox\\matlab\\iofun\\readtable.m',517,0)\">line 517<\/a>)\nUnable to find or open 'emg_normalized_subject6_All.csv'. Check the path and filename or file permissions."}}
%---
