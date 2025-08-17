% Code across all subjects
numSubjects = 40;
accuracy_results = zeros(numSubjects, 4); % Subject Label, Accuracy, Train Time, Test Time

for subject = 1:numSubjects %[output:group:8be768aa]
    fprintf('Processing Subject %d...\n', subject);    %[output:group:8be768aa] %[output:3b8db810] %[output:27e99a43] %[output:32b031aa] %[output:2bdf75d6] %[output:7ea88035] %[output:878d1063] %[output:972967b7] %[output:48378a84] %[output:20001645] %[output:16e755bf] %[output:059398d7] %[output:7ab7c5fd] %[output:9b5040a9] %[output:97f284da] %[output:0470f3db] %[output:55c6058d] %[output:1fe19509] %[output:0a3881cc] %[output:1bf8f46d] %[output:49713171] %[output:2dcbda76] %[output:4ce77ede] %[output:6d626a55] %[output:2d157c6c] %[output:2f044168] %[output:2416666a] %[output:61cb9938] %[output:1eda2fb2] %[output:8cf565ac] %[output:993fdf35] %[output:3b744f35] %[output:4d5db579] %[output:8ad83f78] %[output:7fcfcb56] %[output:6a118f7b] %[output:347d2229] %[output:469cfc18] %[output:4bf30e50] %[output:9ec27b12] %[output:65835616]
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
    %Final model %[output:group:58d655c2]
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
    fprintf('Test Accuracy: %.2f%%\n', accuracy * 100); %[output:5840add8] %[output:647958fd] %[output:4a106c99] %[output:3a26b253] %[output:0b676448] %[output:07a2d1c3] %[output:18b514fa] %[output:0b6947aa] %[output:6e2dbe8f] %[output:5e27691c] %[output:7134dbcd] %[output:557c2a3e] %[output:6f509b30] %[output:3d88819a] %[output:67411432] %[output:3bc7cad3] %[output:32f1ca32] %[output:2e92f515] %[output:3baf814f] %[output:9c212ae1] %[output:7f0e7d6d] %[output:682820f0] %[output:572d3254] %[output:0af6f09d] %[output:302e4b3c] %[output:82b4d69a] %[output:85ba3525] %[output:9031b3fb] %[output:83753912] %[output:01d21e47] %[output:2226e42d] %[output:3b248d68] %[output:40d77a45] %[output:659ab2e1] %[output:59123727] %[output:49b19bce] %[output:4b0f7b95] %[output:2a8a573c] %[output:442210b0] %[output:78ab84ac]
    

    % Store results (subject, accuracy, training time, testing time)
    accuracy_results(subject, :) = [subject, accuracy * 100, training_time, testing_time];

    fprintf('Subject %d - Test Accuracy: %.2f%%\n', subject, accuracy * 100); %[output:4c20ab76] %[output:6e046191] %[output:6d18f0af] %[output:6aadb1b6] %[output:65ccf477] %[output:7426ebfd] %[output:9d02ef4d] %[output:3b36d2f5] %[output:23423d47] %[output:452d0950] %[output:138030ab] %[output:27885709] %[output:3ab134ba] %[output:9e95fd6a] %[output:3c0a6516] %[output:50779f2d] %[output:773d2138] %[output:3d902cec] %[output:884e67c0] %[output:061de142] %[output:21a37ba3] %[output:5d39f456] %[output:1af07ba4] %[output:91945814] %[output:0112b180] %[output:1ee78f41] %[output:15057fd5] %[output:19a6db42] %[output:2664db6a] %[output:6ab6fc06] %[output:952c056f] %[output:6813c232] %[output:69886379] %[output:64886002] %[output:64ae3276] %[output:19fe8f08] %[output:0dc08fe9] %[output:79ea75ed] %[output:61fc8c0e] %[output:04885894]
end %[output:group:58d655c2]

%%
%% Save Accuracy and Time Table
accuracy_table = array2table(accuracy_results, 'VariableNames', {'Subject', 'Accuracy', 'TrainingTime', 'TestingTime'});
disp(accuracy_table); %[output:626e8497]
writetable(accuracy_table, 'SVMAll_Accuracy_and_Times.csv');

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":42}
%---
%[output:3b8db810]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 1...\n","truncated":false}}
%---
%[output:27e99a43]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 2...\n","truncated":false}}
%---
%[output:32b031aa]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 3...\n","truncated":false}}
%---
%[output:2bdf75d6]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 4...\n","truncated":false}}
%---
%[output:7ea88035]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 5...\n","truncated":false}}
%---
%[output:878d1063]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 6...\n","truncated":false}}
%---
%[output:972967b7]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 7...\n","truncated":false}}
%---
%[output:48378a84]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 8...\n","truncated":false}}
%---
%[output:20001645]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 9...\n","truncated":false}}
%---
%[output:16e755bf]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 10...\n","truncated":false}}
%---
%[output:059398d7]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 11...\n","truncated":false}}
%---
%[output:7ab7c5fd]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 12...\n","truncated":false}}
%---
%[output:9b5040a9]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 13...\n","truncated":false}}
%---
%[output:97f284da]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 14...\n","truncated":false}}
%---
%[output:0470f3db]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 15...\n","truncated":false}}
%---
%[output:55c6058d]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 16...\n","truncated":false}}
%---
%[output:1fe19509]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 17...\n","truncated":false}}
%---
%[output:0a3881cc]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 18...\n","truncated":false}}
%---
%[output:1bf8f46d]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 19...\n","truncated":false}}
%---
%[output:49713171]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 20...\n","truncated":false}}
%---
%[output:2dcbda76]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 21...\n","truncated":false}}
%---
%[output:4ce77ede]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 22...\n","truncated":false}}
%---
%[output:6d626a55]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 23...\n","truncated":false}}
%---
%[output:2d157c6c]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 24...\n","truncated":false}}
%---
%[output:2f044168]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 25...\n","truncated":false}}
%---
%[output:2416666a]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 26...\n","truncated":false}}
%---
%[output:61cb9938]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 27...\n","truncated":false}}
%---
%[output:1eda2fb2]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 28...\n","truncated":false}}
%---
%[output:8cf565ac]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 29...\n","truncated":false}}
%---
%[output:993fdf35]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 30...\n","truncated":false}}
%---
%[output:3b744f35]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 31...\n","truncated":false}}
%---
%[output:4d5db579]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 32...\n","truncated":false}}
%---
%[output:8ad83f78]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 33...\n","truncated":false}}
%---
%[output:7fcfcb56]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 34...\n","truncated":false}}
%---
%[output:6a118f7b]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 35...\n","truncated":false}}
%---
%[output:347d2229]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 36...\n","truncated":false}}
%---
%[output:469cfc18]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 37...\n","truncated":false}}
%---
%[output:4bf30e50]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 38...\n","truncated":false}}
%---
%[output:9ec27b12]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 39...\n","truncated":false}}
%---
%[output:65835616]
%   data: {"dataType":"text","outputData":{"text":"Processing Subject 40...\n","truncated":false}}
%---
%[output:5840add8]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 77.86%\n","truncated":false}}
%---
%[output:4c20ab76]
%   data: {"dataType":"text","outputData":{"text":"Subject 1 - Test Accuracy: 77.86%\n","truncated":false}}
%---
%[output:647958fd]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 69.68%\n","truncated":false}}
%---
%[output:6e046191]
%   data: {"dataType":"text","outputData":{"text":"Subject 2 - Test Accuracy: 69.68%\n","truncated":false}}
%---
%[output:4a106c99]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 72.38%\n","truncated":false}}
%---
%[output:6d18f0af]
%   data: {"dataType":"text","outputData":{"text":"Subject 3 - Test Accuracy: 72.38%\n","truncated":false}}
%---
%[output:3a26b253]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 61.83%\n","truncated":false}}
%---
%[output:6aadb1b6]
%   data: {"dataType":"text","outputData":{"text":"Subject 4 - Test Accuracy: 61.83%\n","truncated":false}}
%---
%[output:0b676448]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 80.81%\n","truncated":false}}
%---
%[output:65ccf477]
%   data: {"dataType":"text","outputData":{"text":"Subject 5 - Test Accuracy: 80.81%\n","truncated":false}}
%---
%[output:07a2d1c3]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 74.13%\n","truncated":false}}
%---
%[output:7426ebfd]
%   data: {"dataType":"text","outputData":{"text":"Subject 6 - Test Accuracy: 74.13%\n","truncated":false}}
%---
%[output:18b514fa]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 74.19%\n","truncated":false}}
%---
%[output:9d02ef4d]
%   data: {"dataType":"text","outputData":{"text":"Subject 7 - Test Accuracy: 74.19%\n","truncated":false}}
%---
%[output:0b6947aa]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 81.57%\n","truncated":false}}
%---
%[output:3b36d2f5]
%   data: {"dataType":"text","outputData":{"text":"Subject 8 - Test Accuracy: 81.57%\n","truncated":false}}
%---
%[output:6e2dbe8f]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 76.71%\n","truncated":false}}
%---
%[output:23423d47]
%   data: {"dataType":"text","outputData":{"text":"Subject 9 - Test Accuracy: 76.71%\n","truncated":false}}
%---
%[output:5e27691c]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 70.19%\n","truncated":false}}
%---
%[output:452d0950]
%   data: {"dataType":"text","outputData":{"text":"Subject 10 - Test Accuracy: 70.19%\n","truncated":false}}
%---
%[output:7134dbcd]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 63.84%\n","truncated":false}}
%---
%[output:138030ab]
%   data: {"dataType":"text","outputData":{"text":"Subject 11 - Test Accuracy: 63.84%\n","truncated":false}}
%---
%[output:557c2a3e]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 56.87%\n","truncated":false}}
%---
%[output:27885709]
%   data: {"dataType":"text","outputData":{"text":"Subject 12 - Test Accuracy: 56.87%\n","truncated":false}}
%---
%[output:6f509b30]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 76.74%\n","truncated":false}}
%---
%[output:3ab134ba]
%   data: {"dataType":"text","outputData":{"text":"Subject 13 - Test Accuracy: 76.74%\n","truncated":false}}
%---
%[output:3d88819a]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 59.02%\n","truncated":false}}
%---
%[output:9e95fd6a]
%   data: {"dataType":"text","outputData":{"text":"Subject 14 - Test Accuracy: 59.02%\n","truncated":false}}
%---
%[output:67411432]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 74.55%\n","truncated":false}}
%---
%[output:3c0a6516]
%   data: {"dataType":"text","outputData":{"text":"Subject 15 - Test Accuracy: 74.55%\n","truncated":false}}
%---
%[output:3bc7cad3]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 67.56%\n","truncated":false}}
%---
%[output:50779f2d]
%   data: {"dataType":"text","outputData":{"text":"Subject 16 - Test Accuracy: 67.56%\n","truncated":false}}
%---
%[output:32f1ca32]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 71.99%\n","truncated":false}}
%---
%[output:773d2138]
%   data: {"dataType":"text","outputData":{"text":"Subject 17 - Test Accuracy: 71.99%\n","truncated":false}}
%---
%[output:2e92f515]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 61.97%\n","truncated":false}}
%---
%[output:3d902cec]
%   data: {"dataType":"text","outputData":{"text":"Subject 18 - Test Accuracy: 61.97%\n","truncated":false}}
%---
%[output:3baf814f]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 68.95%\n","truncated":false}}
%---
%[output:884e67c0]
%   data: {"dataType":"text","outputData":{"text":"Subject 19 - Test Accuracy: 68.95%\n","truncated":false}}
%---
%[output:9c212ae1]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 76.90%\n","truncated":false}}
%---
%[output:061de142]
%   data: {"dataType":"text","outputData":{"text":"Subject 20 - Test Accuracy: 76.90%\n","truncated":false}}
%---
%[output:7f0e7d6d]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 68.55%\n","truncated":false}}
%---
%[output:21a37ba3]
%   data: {"dataType":"text","outputData":{"text":"Subject 21 - Test Accuracy: 68.55%\n","truncated":false}}
%---
%[output:682820f0]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 71.98%\n","truncated":false}}
%---
%[output:5d39f456]
%   data: {"dataType":"text","outputData":{"text":"Subject 22 - Test Accuracy: 71.98%\n","truncated":false}}
%---
%[output:572d3254]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 75.08%\n","truncated":false}}
%---
%[output:1af07ba4]
%   data: {"dataType":"text","outputData":{"text":"Subject 23 - Test Accuracy: 75.08%\n","truncated":false}}
%---
%[output:0af6f09d]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 56.13%\n","truncated":false}}
%---
%[output:91945814]
%   data: {"dataType":"text","outputData":{"text":"Subject 24 - Test Accuracy: 56.13%\n","truncated":false}}
%---
%[output:302e4b3c]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 65.36%\n","truncated":false}}
%---
%[output:0112b180]
%   data: {"dataType":"text","outputData":{"text":"Subject 25 - Test Accuracy: 65.36%\n","truncated":false}}
%---
%[output:82b4d69a]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 70.35%\n","truncated":false}}
%---
%[output:1ee78f41]
%   data: {"dataType":"text","outputData":{"text":"Subject 26 - Test Accuracy: 70.35%\n","truncated":false}}
%---
%[output:85ba3525]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 63.51%\n","truncated":false}}
%---
%[output:15057fd5]
%   data: {"dataType":"text","outputData":{"text":"Subject 27 - Test Accuracy: 63.51%\n","truncated":false}}
%---
%[output:9031b3fb]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 59.38%\n","truncated":false}}
%---
%[output:19a6db42]
%   data: {"dataType":"text","outputData":{"text":"Subject 28 - Test Accuracy: 59.38%\n","truncated":false}}
%---
%[output:83753912]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 64.98%\n","truncated":false}}
%---
%[output:2664db6a]
%   data: {"dataType":"text","outputData":{"text":"Subject 29 - Test Accuracy: 64.98%\n","truncated":false}}
%---
%[output:01d21e47]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 72.87%\n","truncated":false}}
%---
%[output:6ab6fc06]
%   data: {"dataType":"text","outputData":{"text":"Subject 30 - Test Accuracy: 72.87%\n","truncated":false}}
%---
%[output:2226e42d]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 63.56%\n","truncated":false}}
%---
%[output:952c056f]
%   data: {"dataType":"text","outputData":{"text":"Subject 31 - Test Accuracy: 63.56%\n","truncated":false}}
%---
%[output:3b248d68]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 66.12%\n","truncated":false}}
%---
%[output:6813c232]
%   data: {"dataType":"text","outputData":{"text":"Subject 32 - Test Accuracy: 66.12%\n","truncated":false}}
%---
%[output:40d77a45]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 80.70%\n","truncated":false}}
%---
%[output:69886379]
%   data: {"dataType":"text","outputData":{"text":"Subject 33 - Test Accuracy: 80.70%\n","truncated":false}}
%---
%[output:659ab2e1]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 69.86%\n","truncated":false}}
%---
%[output:64886002]
%   data: {"dataType":"text","outputData":{"text":"Subject 34 - Test Accuracy: 69.86%\n","truncated":false}}
%---
%[output:59123727]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 62.22%\n","truncated":false}}
%---
%[output:64ae3276]
%   data: {"dataType":"text","outputData":{"text":"Subject 35 - Test Accuracy: 62.22%\n","truncated":false}}
%---
%[output:49b19bce]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 63.85%\n","truncated":false}}
%---
%[output:19fe8f08]
%   data: {"dataType":"text","outputData":{"text":"Subject 36 - Test Accuracy: 63.85%\n","truncated":false}}
%---
%[output:4b0f7b95]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 71.54%\n","truncated":false}}
%---
%[output:0dc08fe9]
%   data: {"dataType":"text","outputData":{"text":"Subject 37 - Test Accuracy: 71.54%\n","truncated":false}}
%---
%[output:2a8a573c]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 60.61%\n","truncated":false}}
%---
%[output:79ea75ed]
%   data: {"dataType":"text","outputData":{"text":"Subject 38 - Test Accuracy: 60.61%\n","truncated":false}}
%---
%[output:442210b0]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 69.60%\n","truncated":false}}
%---
%[output:61fc8c0e]
%   data: {"dataType":"text","outputData":{"text":"Subject 39 - Test Accuracy: 69.60%\n","truncated":false}}
%---
%[output:78ab84ac]
%   data: {"dataType":"text","outputData":{"text":"Test Accuracy: 79.62%\n","truncated":false}}
%---
%[output:04885894]
%   data: {"dataType":"text","outputData":{"text":"Subject 40 - Test Accuracy: 79.62%\n","truncated":false}}
%---
%[output:626e8497]
%   data: {"dataType":"text","outputData":{"text":"    <strong>Subject<\/strong>    <strong>Accuracy<\/strong>    <strong>TrainingTime<\/strong>    <strong>TestingTime<\/strong>\n    <strong>_______<\/strong>    <strong>________<\/strong>    <strong>____________<\/strong>    <strong>___________<\/strong>\n\n       1        77.861        9.3364         2.7052   \n       2        69.678        50.006         43.312   \n       3        72.385        14.432         3.0187   \n       4        61.835         19.64         3.2421   \n       5         80.81        7.9826         2.3348   \n       6        74.135        11.607          2.889   \n       7        74.192        17.623          3.703   \n       8        81.567        10.851         2.5558   \n       9        76.713        11.736         2.9879   \n      10        70.189         14.92         3.2516   \n      11        63.837        23.023         3.7066   \n      12        56.873         18.94         3.4241   \n      13        76.744        10.138         2.6374   \n      14        59.019        20.078         3.6854   \n      15        74.545        12.108         2.7967   \n      16        67.564        14.087         3.0104   \n      17        71.995        18.158         3.4604   \n      18        61.969        20.432         3.2709   \n      19        68.945        19.509         3.6605   \n      20        76.896        11.747          3.085   \n      21        68.549         11.44         2.9097   \n      22        71.975        11.403         2.7374   \n      23        75.079        14.603         3.0758   \n      24        56.134        17.469         3.7716   \n      25        65.359        12.964         3.1521   \n      26        70.347         20.22         4.0395   \n      27        63.513        14.558         3.3455   \n      28        59.382        24.214         4.1493   \n      29         64.98        11.015         3.3708   \n      30        72.867        13.139         3.3336   \n      31        63.558        17.584         4.0114   \n      32        66.124        18.493         3.8785   \n      33        80.701        12.072         3.0453   \n      34         69.86        11.625         3.4406   \n      35        62.223        21.063         4.2905   \n      36        63.851        12.638         3.2703   \n      37        71.536        14.103         3.6363   \n      38        60.607         20.31         4.0325   \n      39        69.604        17.527         4.0701   \n      40        79.615        7.4501          2.382   \n\n","truncated":false}}
%---
