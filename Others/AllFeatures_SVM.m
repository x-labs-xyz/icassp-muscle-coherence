% Code across all subjects
numSubjects = 40;
accuracy_results = zeros(numSubjects, 4); % Subject Label, Accuracy, Train Time, Test Time

for subject = 1:numSubjects %[output:group:2db41eae]
    fprintf('Processing Subject %d...\n', subject);    %[output:group:2db41eae] %[output:367564ae] %[output:6e26183b] %[output:2a00e86b] %[output:3736fe41] %[output:6a98fbf0] %[output:79245f7a] %[output:3f50f531] %[output:1996cb95] %[output:28906cbd] %[output:26722750] %[output:4b541bbd] %[output:1281976f] %[output:26b39592] %[output:19e55649] %[output:46f947d9] %[output:100013c1] %[output:13e54cf2] %[output:3015b14c] %[output:656bfa7b] %[output:52e0f251] %[output:5eaafe75] %[output:17a74544] %[output:3a510c92] %[output:9de2d920] %[output:82f09acc] %[output:06e9ee51] %[output:77c742b8] %[output:59dcbfb4] %[output:7967675c] %[output:3fc26820] %[output:58c0b136] %[output:2c517032] %[output:337ff38c] %[output:7a716fa3] %[output:796a9a58] %[output:047c5d0e] %[output:649a08ec] %[output:3dda0d56] %[output:609a3127] %[output:4913362d]
%%
    % Load EMG Data
    Emg = readtable(['emg_normalized_subject', num2str(subject), '_All.csv']);
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
%%
    %% Define Parameters
    fs = 2000;  % Sampling frequency
    window_length = 0.3 * fs;  % 300ms (window size 600 samples)
    overlap = 0.5 * window_length;  % 50% overlap
    step_size = window_length - overlap;
%%
    %% Feature Extraction
    feature_matrix = [];
    label_vector = [];
    
    for start_idx = 1:step_size:(size(Emg, 1) - window_length + 1)
        
        end_idx = start_idx + window_length - 1;
        window_data = Emg(start_idx:end_idx, :);
        move_label = mode(Movements(start_idx:end_idx));
        rep_label = mode(Rep(start_idx:end_idx));
    
        % Extract Features
        IEMG = sum(abs(window_data)); %Integrated EMG
        VAR = var(window_data); %Variance
        WL = sum(abs(diff(window_data))); %Waveform Length
        SSC = sum(diff(sign(diff(window_data))) ~= 0); %Slope Sign Change
        ZC = sum(diff(sign(window_data)) ~= 0); %Zero Crossing
        WAMP = sum(abs(diff(window_data)) > 0.01);  %Willison Amplitude (Threshold of 0.01)
        MAV = mean(abs(window_data));  % Mean Absolute Value
        MAVS = mean(diff(abs(window_data))); %Mean Absolute Value Slope
        RMS = sqrt(mean(window_data.^2));  % Root Mean Square
       
         % Marginal Discrete Wavelet Transform (mDWT) using 4th level wavelet decomposition
        numChannels = size(window_data, 2);  % Number of EMG channels
        mDWT = zeros(1, numChannels);  % Initialize feature vector
    
        for ch = 1:numChannels
        % Perform 4th level wavelet decomposition on each channel
        [C, ~] = wavedec(window_data(:, ch), 4, 'db4');  
        mDWT(ch) = sum(abs(C));  % Sum of absolute coefficients
        end 
        
        % Histogram of EMG (HEMG) using 10 bins
        HEMG = [];
        for i = 1:size(window_data, 2)
            H = histcounts(window_data(:, i), 10, 'Normalization', 'probability');
            HEMG = [HEMG, H];
        end    
    
        % Autoregressive Coefficients (ARC) using 4th order model
        ARC = [];
        for i = 1:size(window_data, 2)
            a = aryule(window_data(:, i), 4);  
            ARC = [ARC, a(2:end)];  
        end
    
        % Mean Frequency (MNF) and Power Spectrum Ratio (PSR)
        freq_axis = (0:floor(window_length/2)) * (fs / window_length);
        Pxx = abs(fft(window_data)).^2;
        Pxx = Pxx(1:length(freq_axis), :);  
        MNF = sum(freq_axis' .* Pxx) ./ sum(Pxx);  
        PSR = sum(Pxx(1:round(length(freq_axis)*0.5), :)) ./ sum(Pxx);  
    
        % Combine all features
        features = [IEMG, VAR, WL, SSC, ZC, WAMP, MAV, RMS, mDWT, HEMG, MAVS, ARC, MNF, PSR];
        feature_matrix = [feature_matrix; features];
        label_vector = [label_vector; move_label, rep_label];
    end
%%
    %% Split into Training and Testing Sets
    train_idx = ismember(label_vector(:,2), [1, 3, 4, 6]);
    test_idx = ismember(label_vector(:,2), [2, 5]);
    
    X_train = feature_matrix(train_idx, :);
    y_train = label_vector(train_idx, 1);
    X_test = feature_matrix(test_idx, :);
    y_test = label_vector(test_idx, 1);
%%
    %Final model %[output:group:8636d8de]
    bestOrder = 2; % Quadratic order
    bestC = 10;
    
    % Train the SVM 
    template_final = templateSVM(...
        'KernelFunction','polynomial',...       
        'PolynomialOrder',bestOrder,...         
        'KernelScale','auto',...
        'BoxConstraint',bestC,...
        'Standardize',true);
    
    % Train the SVM classifier using fitcecoc
    tic;  % Start timer for training
    svm_model = fitcecoc(X_train, y_train, 'Learners', template_final, 'Coding','onevsall', 'ObservationsIn', 'rows');
    training_time = toc;  % End timer for training

    %% Evaluate the Classifier
    tic;  % Start timer for testing
    y_pred = predict(svm_model, X_test);
    testing_time = toc;  % End timer for testing


    accuracy = sum(y_pred == y_test) / length(y_test);
    fprintf('Test Accuracy: %.2f%%\n', accuracy * 100); %[output:1d24681d] %[output:2db21181] %[output:840d8f26] %[output:2a68a4b1] %[output:42b31902] %[output:69aa99c7] %[output:10f38875] %[output:195dde3c] %[output:16eacaeb] %[output:00d50708] %[output:7ba27892] %[output:930309bc] %[output:4443d5b2] %[output:5bc0be17] %[output:0c3b7007] %[output:459823ad] %[output:04b4e9b5] %[output:909cd703] %[output:59ba104d] %[output:699d0aad] %[output:6c03690b] %[output:41d9977a] %[output:56119526] %[output:850184a5] %[output:466f2360] %[output:017d505c] %[output:87a8a110] %[output:337787ab] %[output:555e939f] %[output:5f059f5e] %[output:11026a78] %[output:7eac8d2e] %[output:8ce47904] %[output:71475fae] %[output:43281ac4] %[output:25505924] %[output:88a53f20] %[output:9d1a7dc7] %[output:26e741a0] %[output:92a6c1f5]
    

    % Store results (subject, accuracy, training time, testing time)
    accuracy_results(subject, :) = [subject, accuracy * 100, training_time, testing_time];

    fprintf('Subject %d - Test Accuracy: %.2f%%\n', subject, accuracy * 100); %[output:7c295973] %[output:9f8a13f3] %[output:87db9389] %[output:07f51af6] %[output:77447a37] %[output:75e680c0] %[output:03c9f86d] %[output:1c16449e] %[output:322c36ed] %[output:381fe51f] %[output:2dffedf6] %[output:07a15c79] %[output:0250745c] %[output:08654015] %[output:05cbf920] %[output:2e2103ed] %[output:08d39673] %[output:079a3a28] %[output:29ae904a] %[output:852fb783] %[output:25a42082] %[output:7b973583] %[output:1822b941] %[output:8ac8041d] %[output:5410697c] %[output:9499ae7e] %[output:97438c77] %[output:5a9b00f1] %[output:470addd0] %[output:1b6b34fa] %[output:2e7fb333] %[output:5a209cc1] %[output:19d97741] %[output:441ce7c2] %[output:5962bb45] %[output:9cca9d9b] %[output:2f831c4c] %[output:3d0449ec] %[output:3f8b782d] %[output:48aa2f68]
end %[output:group:8636d8de]

%%
%% Save Accuracy and Time Table
accuracy_table = array2table(accuracy_results, 'VariableNames', {'Subject', 'Accuracy', 'TrainingTime', 'TestingTime'});
disp(accuracy_table); %[output:54e242a5]
writetable(accuracy_table, 'SVMAll_Accuracy_and_Times.csv');

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":42}
%---
%[output:367564ae]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 1...\n","truncated":false}}
%---
%[output:6e26183b]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 2...\n","truncated":false}}
%---
%[output:2a00e86b]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 3...\n","truncated":false}}
%---
%[output:3736fe41]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 4...\n","truncated":false}}
%---
%[output:6a98fbf0]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 5...\n","truncated":false}}
%---
%[output:79245f7a]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 6...\n","truncated":false}}
%---
%[output:3f50f531]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 7...\n","truncated":false}}
%---
%[output:1996cb95]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 8...\n","truncated":false}}
%---
%[output:28906cbd]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 9...\n","truncated":false}}
%---
%[output:26722750]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 10...\n","truncated":false}}
%---
%[output:4b541bbd]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 11...\n","truncated":false}}
%---
%[output:1281976f]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 12...\n","truncated":false}}
%---
%[output:26b39592]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 13...\n","truncated":false}}
%---
%[output:19e55649]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 14...\n","truncated":false}}
%---
%[output:46f947d9]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 15...\n","truncated":false}}
%---
%[output:100013c1]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 16...\n","truncated":false}}
%---
%[output:13e54cf2]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 17...\n","truncated":false}}
%---
%[output:3015b14c]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 18...\n","truncated":false}}
%---
%[output:656bfa7b]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 19...\n","truncated":false}}
%---
%[output:52e0f251]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 20...\n","truncated":false}}
%---
%[output:5eaafe75]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 21...\n","truncated":false}}
%---
%[output:17a74544]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 22...\n","truncated":false}}
%---
%[output:3a510c92]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 23...\n","truncated":false}}
%---
%[output:9de2d920]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 24...\n","truncated":false}}
%---
%[output:82f09acc]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 25...\n","truncated":false}}
%---
%[output:06e9ee51]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 26...\n","truncated":false}}
%---
%[output:77c742b8]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 27...\n","truncated":false}}
%---
%[output:59dcbfb4]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 28...\n","truncated":false}}
%---
%[output:7967675c]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 29...\n","truncated":false}}
%---
%[output:3fc26820]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 30...\n","truncated":false}}
%---
%[output:58c0b136]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 31...\n","truncated":false}}
%---
%[output:2c517032]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 32...\n","truncated":false}}
%---
%[output:337ff38c]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 33...\n","truncated":false}}
%---
%[output:7a716fa3]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 34...\n","truncated":false}}
%---
%[output:796a9a58]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 35...\n","truncated":false}}
%---
%[output:047c5d0e]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 36...\n","truncated":false}}
%---
%[output:649a08ec]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 37...\n","truncated":false}}
%---
%[output:3dda0d56]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 38...\n","truncated":false}}
%---
%[output:609a3127]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 39...\n","truncated":false}}
%---
%[output:4913362d]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 40...\n","truncated":false}}
%---
%[output:1d24681d]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 77.86%\n","truncated":false}}
%---
%[output:7c295973]
%   data: {"dataType":"text","outputData":{"text":"Subject 1 - Test Accuracy: 77.86%\n","truncated":false}}
%---
%[output:2db21181]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 69.68%\n","truncated":false}}
%---
%[output:9f8a13f3]
%   data: {"dataType":"text","outputData":{"text":"Subject 2 - Test Accuracy: 69.68%\n","truncated":false}}
%---
%[output:840d8f26]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 72.38%\n","truncated":false}}
%---
%[output:87db9389]
%   data: {"dataType":"text","outputData":{"text":"Subject 3 - Test Accuracy: 72.38%\n","truncated":false}}
%---
%[output:2a68a4b1]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 61.83%\n","truncated":false}}
%---
%[output:07f51af6]
%   data: {"dataType":"text","outputData":{"text":"Subject 4 - Test Accuracy: 61.83%\n","truncated":false}}
%---
%[output:42b31902]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 80.81%\n","truncated":false}}
%---
%[output:77447a37]
%   data: {"dataType":"text","outputData":{"text":"Subject 5 - Test Accuracy: 80.81%\n","truncated":false}}
%---
%[output:69aa99c7]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 74.13%\n","truncated":false}}
%---
%[output:75e680c0]
%   data: {"dataType":"text","outputData":{"text":"Subject 6 - Test Accuracy: 74.13%\n","truncated":false}}
%---
%[output:10f38875]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 74.19%\n","truncated":false}}
%---
%[output:03c9f86d]
%   data: {"dataType":"text","outputData":{"text":"Subject 7 - Test Accuracy: 74.19%\n","truncated":false}}
%---
%[output:195dde3c]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 81.57%\n","truncated":false}}
%---
%[output:1c16449e]
%   data: {"dataType":"text","outputData":{"text":"Subject 8 - Test Accuracy: 81.57%\n","truncated":false}}
%---
%[output:16eacaeb]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 76.71%\n","truncated":false}}
%---
%[output:322c36ed]
%   data: {"dataType":"text","outputData":{"text":"Subject 9 - Test Accuracy: 76.71%\n","truncated":false}}
%---
%[output:00d50708]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 70.19%\n","truncated":false}}
%---
%[output:381fe51f]
%   data: {"dataType":"text","outputData":{"text":"Subject 10 - Test Accuracy: 70.19%\n","truncated":false}}
%---
%[output:7ba27892]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 63.84%\n","truncated":false}}
%---
%[output:2dffedf6]
%   data: {"dataType":"text","outputData":{"text":"Subject 11 - Test Accuracy: 63.84%\n","truncated":false}}
%---
%[output:930309bc]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 56.87%\n","truncated":false}}
%---
%[output:07a15c79]
%   data: {"dataType":"text","outputData":{"text":"Subject 12 - Test Accuracy: 56.87%\n","truncated":false}}
%---
%[output:4443d5b2]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 76.74%\n","truncated":false}}
%---
%[output:0250745c]
%   data: {"dataType":"text","outputData":{"text":"Subject 13 - Test Accuracy: 76.74%\n","truncated":false}}
%---
%[output:5bc0be17]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 59.02%\n","truncated":false}}
%---
%[output:08654015]
%   data: {"dataType":"text","outputData":{"text":"Subject 14 - Test Accuracy: 59.02%\n","truncated":false}}
%---
%[output:0c3b7007]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 74.55%\n","truncated":false}}
%---
%[output:05cbf920]
%   data: {"dataType":"text","outputData":{"text":"Subject 15 - Test Accuracy: 74.55%\n","truncated":false}}
%---
%[output:459823ad]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 67.56%\n","truncated":false}}
%---
%[output:2e2103ed]
%   data: {"dataType":"text","outputData":{"text":"Subject 16 - Test Accuracy: 67.56%\n","truncated":false}}
%---
%[output:04b4e9b5]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 71.99%\n","truncated":false}}
%---
%[output:08d39673]
%   data: {"dataType":"text","outputData":{"text":"Subject 17 - Test Accuracy: 71.99%\n","truncated":false}}
%---
%[output:909cd703]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 61.97%\n","truncated":false}}
%---
%[output:079a3a28]
%   data: {"dataType":"text","outputData":{"text":"Subject 18 - Test Accuracy: 61.97%\n","truncated":false}}
%---
%[output:59ba104d]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 68.95%\n","truncated":false}}
%---
%[output:29ae904a]
%   data: {"dataType":"text","outputData":{"text":"Subject 19 - Test Accuracy: 68.95%\n","truncated":false}}
%---
%[output:699d0aad]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 76.90%\n","truncated":false}}
%---
%[output:852fb783]
%   data: {"dataType":"text","outputData":{"text":"Subject 20 - Test Accuracy: 76.90%\n","truncated":false}}
%---
%[output:6c03690b]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 68.55%\n","truncated":false}}
%---
%[output:25a42082]
%   data: {"dataType":"text","outputData":{"text":"Subject 21 - Test Accuracy: 68.55%\n","truncated":false}}
%---
%[output:41d9977a]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 71.98%\n","truncated":false}}
%---
%[output:7b973583]
%   data: {"dataType":"text","outputData":{"text":"Subject 22 - Test Accuracy: 71.98%\n","truncated":false}}
%---
%[output:56119526]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 75.08%\n","truncated":false}}
%---
%[output:1822b941]
%   data: {"dataType":"text","outputData":{"text":"Subject 23 - Test Accuracy: 75.08%\n","truncated":false}}
%---
%[output:850184a5]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 56.13%\n","truncated":false}}
%---
%[output:8ac8041d]
%   data: {"dataType":"text","outputData":{"text":"Subject 24 - Test Accuracy: 56.13%\n","truncated":false}}
%---
%[output:466f2360]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 65.36%\n","truncated":false}}
%---
%[output:5410697c]
%   data: {"dataType":"text","outputData":{"text":"Subject 25 - Test Accuracy: 65.36%\n","truncated":false}}
%---
%[output:017d505c]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 70.35%\n","truncated":false}}
%---
%[output:9499ae7e]
%   data: {"dataType":"text","outputData":{"text":"Subject 26 - Test Accuracy: 70.35%\n","truncated":false}}
%---
%[output:87a8a110]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 63.51%\n","truncated":false}}
%---
%[output:97438c77]
%   data: {"dataType":"text","outputData":{"text":"Subject 27 - Test Accuracy: 63.51%\n","truncated":false}}
%---
%[output:337787ab]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 59.38%\n","truncated":false}}
%---
%[output:5a9b00f1]
%   data: {"dataType":"text","outputData":{"text":"Subject 28 - Test Accuracy: 59.38%\n","truncated":false}}
%---
%[output:555e939f]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 64.98%\n","truncated":false}}
%---
%[output:470addd0]
%   data: {"dataType":"text","outputData":{"text":"Subject 29 - Test Accuracy: 64.98%\n","truncated":false}}
%---
%[output:5f059f5e]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 72.87%\n","truncated":false}}
%---
%[output:1b6b34fa]
%   data: {"dataType":"text","outputData":{"text":"Subject 30 - Test Accuracy: 72.87%\n","truncated":false}}
%---
%[output:11026a78]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 63.56%\n","truncated":false}}
%---
%[output:2e7fb333]
%   data: {"dataType":"text","outputData":{"text":"Subject 31 - Test Accuracy: 63.56%\n","truncated":false}}
%---
%[output:7eac8d2e]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 66.12%\n","truncated":false}}
%---
%[output:5a209cc1]
%   data: {"dataType":"text","outputData":{"text":"Subject 32 - Test Accuracy: 66.12%\n","truncated":false}}
%---
%[output:8ce47904]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 80.70%\n","truncated":false}}
%---
%[output:19d97741]
%   data: {"dataType":"text","outputData":{"text":"Subject 33 - Test Accuracy: 80.70%\n","truncated":false}}
%---
%[output:71475fae]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 69.86%\n","truncated":false}}
%---
%[output:441ce7c2]
%   data: {"dataType":"text","outputData":{"text":"Subject 34 - Test Accuracy: 69.86%\n","truncated":false}}
%---
%[output:43281ac4]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 62.22%\n","truncated":false}}
%---
%[output:5962bb45]
%   data: {"dataType":"text","outputData":{"text":"Subject 35 - Test Accuracy: 62.22%\n","truncated":false}}
%---
%[output:25505924]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 63.85%\n","truncated":false}}
%---
%[output:9cca9d9b]
%   data: {"dataType":"text","outputData":{"text":"Subject 36 - Test Accuracy: 63.85%\n","truncated":false}}
%---
%[output:88a53f20]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 71.54%\n","truncated":false}}
%---
%[output:2f831c4c]
%   data: {"dataType":"text","outputData":{"text":"Subject 37 - Test Accuracy: 71.54%\n","truncated":false}}
%---
%[output:9d1a7dc7]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 60.61%\n","truncated":false}}
%---
%[output:3d0449ec]
%   data: {"dataType":"text","outputData":{"text":"Subject 38 - Test Accuracy: 60.61%\n","truncated":false}}
%---
%[output:26e741a0]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 69.60%\n","truncated":false}}
%---
%[output:3f8b782d]
%   data: {"dataType":"text","outputData":{"text":"Subject 39 - Test Accuracy: 69.60%\n","truncated":false}}
%---
%[output:92a6c1f5]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 79.62%\n","truncated":false}}
%---
%[output:48aa2f68]
%   data: {"dataType":"text","outputData":{"text":"Subject 40 - Test Accuracy: 79.62%\n","truncated":false}}
%---
%[output:54e242a5]
%   data: {"dataType":"text","outputData":{"text":"    <strong>Subject<\/strong>    <strong>Accuracy<\/strong>    <strong>TrainingTime<\/strong>    <strong>TestingTime<\/strong>\n    <strong>_______<\/strong>    <strong>________<\/strong>    <strong>____________<\/strong>    <strong>___________<\/strong>\n\n       1        77.861        9.3364         2.7052   \n       2        69.678        50.006         43.312   \n       3        72.385        14.432         3.0187   \n       4        61.835         19.64         3.2421   \n       5         80.81        7.9826         2.3348   \n       6        74.135        11.607          2.889   \n       7        74.192        17.623          3.703   \n       8        81.567        10.851         2.5558   \n       9        76.713        11.736         2.9879   \n      10        70.189         14.92         3.2516   \n      11        63.837        23.023         3.7066   \n      12        56.873         18.94         3.4241   \n      13        76.744        10.138         2.6374   \n      14        59.019        20.078         3.6854   \n      15        74.545        12.108         2.7967   \n      16        67.564        14.087         3.0104   \n      17        71.995        18.158         3.4604   \n      18        61.969        20.432         3.2709   \n      19        68.945        19.509         3.6605   \n      20        76.896        11.747          3.085   \n      21        68.549         11.44         2.9097   \n      22        71.975        11.403         2.7374   \n      23        75.079        14.603         3.0758   \n      24        56.134        17.469         3.7716   \n      25        65.359        12.964         3.1521   \n      26        70.347         20.22         4.0395   \n      27        63.513        14.558         3.3455   \n      28        59.382        24.214         4.1493   \n      29         64.98        11.015         3.3708   \n      30        72.867        13.139         3.3336   \n      31        63.558        17.584         4.0114   \n      32        66.124        18.493         3.8785   \n      33        80.701        12.072         3.0453   \n      34         69.86        11.625         3.4406   \n      35        62.223        21.063         4.2905   \n      36        63.851        12.638         3.2703   \n      37        71.536        14.103         3.6363   \n      38        60.607         20.31         4.0325   \n      39        69.604        17.527         4.0701   \n      40        79.615        7.4501          2.382   \n\n","truncated":false}}
%---
