% Code to export the MSC matrices for all subjects, movements and
% repetitions.

%Last modified: 17/03/205 by Costanza Armanini

%%
%[text] ## Initial settings
% Number of movements and repetitions

numSubjects = 40; % Number of subjects
numMovements = 49; % Number of movements
numReps = 6; % Number of repetitions
numSensors = 12; % Number of sensors

% Parameters for Welch Method

window = 600; % window size for Welch method
noverlap = window / 2; % 50% overlap
Fs = 2000; % sampling frequency in Hz
    
%%
%[text] ## For ycle across all subjects
for subject=1:numSubjects


    %create folder for the subject ID
    dir_name = sprintf('Subject%d', subject); 
    mkdir(dir_name);
    
    
    %% Load emg signals, which have been already filtered and normalized, with movement and repetition labels
    Emg = readtable(['data/emg_normalized_subject', num2str(subject), '_All.csv']);
    Emg(1,:) = [];
    Emg(:,1) = [];
    Emg = Emg{:,:};
    
    Movements = readtable(['data/move_labels_subject', num2str(subject), '_All.csv']);
    Movements(1,:) = [];
    Movements(:,1) = [];
    Movements = Movements{:,:};
    
    Rep = readtable(['data/rep_labels_subject', num2str(subject), '_All.csv']);
    Rep(1,:) = [];
    Rep(:,1) = [];
    Rep = Rep{:,:};
    
     
    % Create a cell array to hold data for each movement and repetition
    MovementData = cell(numMovements, numReps);
    
    % Loop over all movements and repetitions to fill the cell array
    for mov = 1:numMovements
        for rep = 1:numReps
            % Find indices where the movement and repetition match
            idx = Movements == mov & Rep == rep;
            % Extract corresponding rows from Emg
            MovementData{mov, rep} = Emg(idx, :);
        end
    end
    
    %% Initialize cell arrays to store the PSD, CPSD, and MSC
    PSD = cell(numMovements, numReps, numSensors);
    CPSD = cell(numMovements, numReps, numSensors, numSensors);
    MSC = cell(numMovements, numReps, numSensors, numSensors);
    
    %% Analyze each movement and repetition
    for mov = 1:numMovements
        for rep = 1:numReps
            for s1 = 1:numSensors
                % Calculate PSD for each sensor
                PSD{mov, rep, s1} = pwelch(MovementData{mov, rep}(:, s1), window, noverlap, [], Fs);
    
                % Set the diagonal values of MSC to 1
                MSC{mov, rep, s1, s1} = 1;
    
                for s2 = 1:numSensors
                    if s1 ~= s2 % Only calculate CPSD and MSC if sensors are different
                        % Calculate CPSD and MSC between each pair of sensors
                        [cpsd_val, freq] = cpsd(MovementData{mov, rep}(:, s1), MovementData{mov, rep}(:, s2), window, noverlap, [], Fs);
                        msc_val = mscohere(MovementData{mov, rep}(:, s1), MovementData{mov, rep}(:, s2), window, noverlap, [], Fs);
    
                        CPSD{mov, rep, s1, s2} = cpsd_val;
                        MSC{mov, rep, s1, s2} = msc_val;
    
                        % Since CPSD and MSC are symmetric properties, mirror them across the diagonal
                        CPSD{mov, rep, s2, s1} = cpsd_val;
                        MSC{mov, rep, s2, s1} = msc_val;
                    end
                end
            end
        end
    end
    
    % Save and export for each movement
    nFreqs = length(MSC{1, 1, 1, 2}); % Number of frequencies, assumed same for all non-diagonal elements
    
    % Loop through each movement
    for mov = 1:numMovements
        % Initialize a cell array to hold all repetition data for this movement
        MSC_allReps = cell(numReps, 1);
        
        % Gather data for each repetition
        for rep = 1:numReps
            % Prepare a matrix to store MSC values for this movement and repetition
            MSC_matrix = zeros(numSensors, numSensors, nFreqs); % Correctly initialized to handle frequency dimension
            
            % Fill the MSC_matrix with data
            for s1 = 1:numSensors
                for s2 = 1:numSensors
                    if s1 == s2
                        % Set the entire frequency dimension to 1 for diagonal elements
                        MSC_matrix(s1, s2, :) = 1;
                    else
                        % Ensure the MSC data is assigned correctly
                        MSC_matrix(s1, s2, :) = reshape(MSC{mov, rep, s1, s2}, [1, 1, nFreqs]);
                    end
                end
            end
            
            % Store the matrix in the cell array
            MSC_allReps{rep} = MSC_matrix;
        end
        
        % Check for recurring MSC matrices among the repetitions
        mscDuplicates = false(numReps, numReps);  % Initialize matrix to store MSC duplicates info
    
        for i = 1:numReps
            for j = i+1:numReps  % Check only once for each pair
                if isequal(MSC_allReps{i}, MSC_allReps{j})  % Compare MSC matrices
                    mscDuplicates(i, j) = true;
                    mscDuplicates(j, i) = true;  % Symmetric check
                end
            end
        end
    
        % Display duplicate MSC matrices (if any)
        [row, col] = find(mscDuplicates);
    
        if ~isempty(row)
            disp(['Duplicate MSC matrices found for Movement ', num2str(mov)]);
            for k = 1:numel(row)
                fprintf('Repetition %d and Repetition %d have identical MSC matrices.\n', row(k), col(k));
            end
            disp('Stopping execution due to duplicate MSC matrices.');
            return;  % Stop execution if duplicate MSC matrices are found
        else
            disp(['No duplicate MSC matrices found for Movement ', num2str(mov)]);
        end
    
        % File name for saving, incorporating just the movement number
        filename = fullfile(dir_name, sprintf('MSC_S%d_mov%d.mat', subject, mov));
        
        % Save the cell array to a .mat file
        save(filename, 'MSC_allReps');
    end

end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":5.8}
%---
