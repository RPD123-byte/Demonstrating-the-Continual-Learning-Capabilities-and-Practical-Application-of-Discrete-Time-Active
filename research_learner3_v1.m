load('all_variables_research_learner.mat');

figureFolder = 'figures4';

% Define new outcomes for each industry
predefined_outcomes = cell(16, 4);

% Modify some of the predefined outcomes
for f1 = 1:16
    if f1 == 1 % Food industry
        predefined_outcomes(f1,:) = {'Good', 'Terrible', 'Neutral', 'Excellent'};
    elseif f1 == 2 % Oil industry
        predefined_outcomes(f1,:) = {'Terrible', 'Bad', 'Excellent', 'Neutral'};
    elseif f1 == 3 % Gas industry
        predefined_outcomes(f1,:) = {'Good', 'Neutral', 'Terrible', 'Bad'};
    elseif f1 == 4 % Nuclear industry
        predefined_outcomes(f1,:) = {'Neutral', 'Good', 'Bad', 'Terrible'};
    elseif f1 == 5 % Chemical industry
        predefined_outcomes(f1,:) = {'Bad', 'Good', 'Terrible', 'Neutral'};
    elseif f1 == 6 % Automotive industry
        predefined_outcomes(f1,:) = {'Neutral', 'Terrible', 'Good', 'Excellent'};
    elseif f1 == 7 % Aerospace industry
        predefined_outcomes(f1,:) = {'Terrible', 'Excellent', 'Neutral', 'Good'};
    elseif f1 == 8 % Telecommunications industry
        predefined_outcomes(f1,:) = {'Good', 'Neutral', 'Excellent', 'Bad'};
    elseif f1 == 9 % Electronics industry
        predefined_outcomes(f1,:) = {'Excellent', 'Terrible', 'Good', 'Neutral'};
    elseif f1 == 10 % Textile industry
        predefined_outcomes(f1,:) = {'Neutral', 'Bad', 'Excellent', 'Terrible'};
    elseif f1 == 11 % Pharmaceutical industry
        predefined_outcomes(f1,:) = {'Bad', 'Excellent', 'Terrible', 'Good'};
    elseif f1 == 12 % Construction industry
        predefined_outcomes(f1,:) = {'Neutral', 'Good', 'Terrible', 'Excellent'};
    elseif f1 == 13 % Agriculture industry
        predefined_outcomes(f1,:) = {'Terrible', 'Neutral', 'Good', 'Bad'};
    elseif f1 == 14 % Mining industry
        predefined_outcomes(f1,:) = {'Good', 'Neutral', 'Bad', 'Terrible'};
    elseif f1 == 15 % Renewable energy industry
        predefined_outcomes(f1,:) = {'Neutral', 'Terrible', 'Good', 'Excellent'};
    elseif f1 == 16 % Transport industry
        predefined_outcomes(f1,:) = {'Terrible', 'Neutral', 'Excellent', 'Good'};
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

% Update the A matrices inthe AfterSim structure
AfterSim.MDP.A = A;
AfterSim.MDP.eta = 0.1;

% The lowercase 'a' matrices remain unchanged
% AfterSim.MDP.a = AfterSim.MDP.a;

% Save the updated AfterSim structure
save('modified_agent_x.mat', 'AfterSim');

% Optional: Display a message to confirm the update
disp('Environment modified and AfterSim structure updated.');

N_new_new = 20;
% illustrate a single trial
%==========================================================================
mdp = AfterSim;
BeforeSim = mdp; % before simulations

load('scores_data.mat');

scores_all_new = [scores_all zeros(1, N_new_new)]; % Extend the scores array

% Run active inference agent for N iterations
for i = 1:N_new_new
    MMDP(N + i) = spm_MDP_VB_X(mdp);
    % Update the a matrix with the learned probabilities
    for j = 1:numel(mdp.MDP.a)
        mdp.MDP.a{j} = MMDP(N + i).mdp(end).a{j};
    end

    % Calculate the score for all 16 industries
    score_all = 0;
    fprintf('Iteration %d:\n', N + i);
    for industry = 1:16
        for process = 1:4
            % Get the values in the lowercase 'a' matrix
            a_values = mdp.MDP.a{2}(:, industry, process);
            % Get the correct outcome for this process
            correct_outcome = predefined_outcomes{industry, process};
            % Map the outcome to an index
            outcome_map = containers.Map({'Excellent', 'Good', 'Neutral', 'Bad', 'Terrible'}, 1:5);
            correct_index = outcome_map(correct_outcome);
            % Calculate process score
            correct_belief = a_values(correct_index);
            incorrect_belief = max(a_values([1:correct_index-1, correct_index+1:end]));
            process_score = (correct_belief - incorrect_belief + 1) / 2; % Normalize to [0, 1]
            score_all = score_all + process_score;
        end
    end
    scores_all_new(N + i) = score_all;
    fprintf('  Total Score (All Industries) = %.4f\n\n', score_all);
end

% Plot the score over time including the new iterations
figure;
plot(1:(N + N_new_new), scores_all_new, '-o');
xlabel('Iteration');
ylabel('Score');
title('Learning Progress of the Active Inference Agent (All Industries)');
grid on;
saveas(gcf, fullfile(figureFolder, 'learning_progress_all_industries_new.png'));

AfterSim = mdp; % after simulations

% save('all_variables_research_learner3_v1.mat');
