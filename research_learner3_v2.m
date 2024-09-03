load('/Users/rithvikprakki/ARC-AGI/all_variables_research_learner.mat');
load('/Users/rithvikprakki/ARC-AGI/all_variables_research_learner2.mat');

figureFolder = '/Users/rithvikprakki/ARC-AGI/figures6';

% Define new outcomes for each industry
predefined_outcomes = cell(16, 4);

% Modify some of the predefined outcomes
for f1 = 1:16
    if f1 == 1 % Food industry
        predefined_outcomes(f1,:) = {'Excellent', 'Neutral', 'Good', 'Bad'};
    elseif f1 == 2 % Oil industry
        predefined_outcomes(f1,:) = {'Bad', 'Neutral', 'Terrible', 'Good'};
    elseif f1 == 3 % Gas industry
        predefined_outcomes(f1,:) = {'Neutral', 'Bad', 'Bad', 'Excellent'};
    elseif f1 == 4 % Nuclear industry
        predefined_outcomes(f1,:) = {'Good', 'Terrible', 'Good', 'Neutral'};
    elseif f1 == 5 % Chemical industry
        predefined_outcomes(f1,:) = {'Terrible', 'Neutral', 'Bad', 'Good'};
    elseif f1 == 6 % Automotive industry
        predefined_outcomes(f1,:) = {'Excellent', 'Excellent', 'Neutral', 'Good'};
    elseif f1 == 7 % Aerospace industry
        predefined_outcomes(f1,:) = {'Neutral', 'Good', 'Terrible', 'Bad'};
    elseif f1 == 8 % Telecommunications industry
        predefined_outcomes(f1,:) = {'Bad', 'Good', 'Neutral', 'Terrible'};
    elseif f1 == 9 % Electronics industry
        predefined_outcomes(f1,:) = {'Neutral', 'Excellent', 'Neutral', 'Good'};
    elseif f1 == 10 % Textile industry
        predefined_outcomes(f1,:) = {'Terrible', 'Good', 'Excellent', 'Neutral'};
    elseif f1 == 11 % Pharmaceutical industry
        predefined_outcomes(f1,:) = {'Excellent', 'Bad', 'Neutral', 'Good'};
    elseif f1 == 12 % Construction industry
        predefined_outcomes(f1,:) = {'Terrible', 'Neutral', 'Terrible', 'Bad'};
    elseif f1 == 13 % Agriculture industry
        predefined_outcomes(f1,:) = {'Neutral', 'Good', 'Bad', 'Excellent'};
    elseif f1 == 14 % Mining industry
        predefined_outcomes(f1,:) = {'Bad', 'Neutral', 'Good', 'Terrible'};
    elseif f1 == 15 % Renewable energy industry
        predefined_outcomes(f1,:) = {'Good', 'Terrible', 'Excellent', 'Neutral'};
    elseif f1 == 16 % Transport industry
        predefined_outcomes(f1,:) = {'Neutral', 'Excellent', 'Terrible', 'Good'};
    end
end

d{1} = ones(16,1);   % context: industry identity 
d{2} = zeros(4,1);  % research process: there are 15 processes
d{2}(1) = 1;         % start from the first research process

Nf = numel(d);
for f = 1:Nf
    Ns(f) = numel(d{f});
end

% Initialize new A matrices
A = AfterSim.MDP.A;
A{2} = zeros(size(AfterSim.MDP.A{2})); % Reset A{2} as it will be completely rebuilt

% Update A matrices based on new predefined outcomes
for f1 = 1:Ns(1) % context: industry identity
    for f2 = 1:Ns(2) % research process
        % Use the predefined outcome for this industry and process
        currentOutcome = predefined_outcomes{f1,f2};
        
        % A{1} remains unchanged
        % A{1}(f2,f1,f2) = 1; 
        
        % Update A{2}
        A{2}(1,f1,f2) = strcmp(currentOutcome,'Excellent');
        A{2}(2,f1,f2) = strcmp(currentOutcome,'Good');
        A{2}(3,f1,f2) = strcmp(currentOutcome,'Neutral');
        A{2}(4,f1,f2) = strcmp(currentOutcome,'Bad');
        A{2}(5,f1,f2) = strcmp(currentOutcome,'Terrible');
        
        % A{3} remains unchanged
        % A{3}(f1,f1,f2) = 1;
    end
end

% Update the A matrices in the AfterSim structure
AfterSim.MDP.A = A;

% The lowercase 'a' matrices remain unchanged
% AfterSim.MDP.a = AfterSim.MDP.a;

% Save the updated AfterSim structure
save('modified_agent_x.mat', 'AfterSim');

% Optional: Display a message to confirm the update
disp('Environment modified and AfterSim structure updated.');

% OPTIONS.figureFolder = "/Users/rithvikprakki/ARC-AGI/figures5";

N_new_new = 2;
% illustrate a single trial
%==========================================================================
mdp = AfterSim;
BeforeSim = mdp; % before simulations

scores = [scores zeros(1, N_new_new)]; % Extend the scores array

% Run active inference agent for N iterations
for i = 1:N_new_new
    MMDP(N + i) = spm_MDP_VB_X(mdp);
    
    % Update the a matrix with the learned probabilities
    for j = 1:numel(mdp.MDP.a)
        mdp.MDP.a{j} = MMDP(N + i).mdp(end).a{j};
    end
    
    % Calculate the score for this iteration
    score = 0;
    for industry = 1:16
        for process = 1:4
            % Get the values in the lowercase 'a' matrix
            a_value1 = mdp.MDP.a{2}(1, industry, process);
            a_value2 = mdp.MDP.a{2}(2, industry, process);
            a_value3 = mdp.MDP.a{2}(3, industry, process);
            a_value4 = mdp.MDP.a{2}(4, industry, process);
            a_value5 = mdp.MDP.a{2}(5, industry, process);

            % Get the values in the uppercase 'A' matrix
            A_value1 = mdp.MDP.A{2}(1, industry, process);
            A_value2 = mdp.MDP.A{2}(2, industry, process);
            A_value3 = mdp.MDP.A{2}(3, industry, process);
            A_value4 = mdp.MDP.A{2}(4, industry, process);
            A_value5 = mdp.MDP.A{2}(5, industry, process);

            % Add the product of a_value and A_value to the score
            if A_value1 ~= 0
                score = score + (a_value1 * A_value1);
            end
            if A_value2 ~= 0
                score = score + (a_value2 * A_value2);
            end
            if A_value3 ~= 0
                score = score + (a_value3 * A_value3);
            end
            if A_value4 ~= 0
                score = score + (a_value4 * A_value4);
            end
            if A_value5 ~= 0
                score = score + (a_value5 * A_value5);
            end
        end
    end
    scores(N + N_new + i) = score; % Store the score for this iteration
end

% Plot the score over time including the new iterations
figure;
plot(1:(N + N_new + N_new_new), scores, '-o');
xlabel('Iteration');
ylabel('Score');
title('Learning Progress of the Active Inference Agent');
grid on;
saveas(gcf, fullfile(figureFolder, 'learning_progress.png'));

AfterSim = mdp; % after simulations


% Function to display and save a matrix or tensor
function display_matrix(data, title_str, figureFolder)
    F = spm_figure('Create', title_str);
    
    if isnumeric(data)
        dims = ndims(data);
        if dims <= 2
            % It's a matrix, display it directly
            imagesc(data);
            colorbar;
            title(title_str);
        else
            % It's a tensor, display each slice
            slices = size(data, 3);
            rows = ceil(sqrt(slices));
            cols = ceil(slices / rows);
            for i = 1:slices
                subplot(rows, cols, i);
                imagesc(data(:,:,i));
                colorbar;
                title(sprintf('Slice %d', i));
            end
            sgtitle(title_str);
        end
    elseif iscell(data)
        % If it's a cell array, display each cell
        num_cells = numel(data);
        rows = ceil(sqrt(num_cells));
        cols = ceil(num_cells / rows);
        for i = 1:num_cells
            subplot(rows, cols, i);
            if isnumeric(data{i})
                imagesc(data{i});
                colorbar;
            else
                text(0.5, 0.5, sprintf('Non-numeric data: %s', class(data{i})), 'HorizontalAlignment', 'center');
            end
            title(sprintf('Cell %d', i));
        end
        sgtitle(title_str);
    else
        % If it's neither numeric nor a cell array, display an error message
        text(0.5, 0.5, sprintf('Unable to display data of type: %s', class(data)), 'HorizontalAlignment', 'center');
        title(title_str);
    end
    
    colormap(gray);
    
    % Save the figure
    fileName = fullfile(figureFolder, [strrep(title_str, ' ', '_'), '.png']);
    saveas(F, fileName);
    
    % Close the figure to free up memory
    close(F);
end

% Display matrices/tensors for BeforeSim
disp('Displaying and saving matrices/tensors for BeforeSim:');

% Level 1 (MDP)
disp('Level 1 (MDP):');
for i = 1:length(BeforeSim.MDP.a)
    display_matrix(BeforeSim.MDP.a{i}, sprintf('BeforeSim_Level1_littlea%d', i), figureFolder);
end
for i = 1:length(BeforeSim.MDP.A)
    display_matrix(BeforeSim.MDP.A{i}, sprintf('BeforeSim_Level1_bigA%d', i), figureFolder);
end
% for i = 1:length(BeforeSim.MDP.B)
%     display_matrix(BeforeSim.MDP.B{i}, sprintf('BeforeSim_Level1_B%d', i), figureFolder);
% end
% for i = 1:length(BeforeSim.MDP.C)
%     display_matrix(BeforeSim.MDP.C{i}, sprintf('BeforeSim_Level1_C%d', i), figureFolder);
% end
% display_matrix(BeforeSim.MDP.D, 'BeforeSim_Level1_D', figureFolder);
% display_matrix(BeforeSim.MDP.V, 'BeforeSim_Level1_V', figureFolder);

% Level 2
disp('Level 2:');
for i = 1:length(BeforeSim.A)
    display_matrix(BeforeSim.A{i}, sprintf('BeforeSim_Level2_A%d', i), figureFolder);
end
% for i = 1:length(BeforeSim.B)
%     display_matrix(BeforeSim.B{i}, sprintf('BeforeSim_Level2_B%d', i), figureFolder);
% end
% for i = 1:length(BeforeSim.C)
%     display_matrix(BeforeSim.C{i}, sprintf('BeforeSim_Level2_C%d', i), figureFolder);
% end
% display_matrix(BeforeSim.D, 'BeforeSim_Level2_D', figureFolder);
% display_matrix(BeforeSim.U, 'BeforeSim_Level2_U', figureFolder);

% Display matrices/tensors for AfterSim
disp('Displaying and saving matrices/tensors for AfterSim:');

% Level 1 (MDP)
disp('Level 1 (MDP):');
for i = 1:length(AfterSim.MDP.a)
    display_matrix(AfterSim.MDP.a{i}, sprintf('AfterSim_Level1_littlea%d', i), figureFolder);
end
for i = 1:length(AfterSim.MDP.A)
    display_matrix(AfterSim.MDP.A{i}, sprintf('AfterSim_Level1_bigA%d', i), figureFolder);
end
% for i = 1:length(AfterSim.MDP.B)
%     display_matrix(AfterSim.MDP.B{i}, sprintf('AfterSim_Level1_B%d', i), figureFolder);
% end
% for i = 1:length(AfterSim.MDP.C)
%     display_matrix(AfterSim.MDP.C{i}, sprintf('AfterSim_Level1_C%d', i), figureFolder);
% end
% display_matrix(AfterSim.MDP.D, 'AfterSim_Level1_D', figureFolder);
% display_matrix(AfterSim.MDP.V, 'AfterSim_Level1_V', figureFolder);

% Level 2
disp('Level 2:');
for i = 1:length(AfterSim.A)
    display_matrix(AfterSim.A{i}, sprintf('AfterSim_Level2_A%d', i), figureFolder);
end
% for i = 1:length(AfterSim.B)
%     display_matrix(AfterSim.B{i}, sprintf('AfterSim_Level2_B%d', i), figureFolder);
% end
% for i = 1:length(AfterSim.C)
%     display_matrix(AfterSim.C{i}, sprintf('AfterSim_Level2_C%d', i), figureFolder);
% end
% display_matrix(AfterSim.D, 'AfterSim_Level2_D', figureFolder);
% display_matrix(AfterSim.U, 'AfterSim_Level2_U', figureFolder);

disp('All matrices/tensors have been displayed and saved.');

save('/Users/rithvikprakki/ARC-AGI/all_variables_research_learner3_v2.mat');
