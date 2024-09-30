load('all_variables_research_learner.mat');

figureFolder = 'figures3';

% Define new outcomes for each industry
predefined_outcomes_1 = get_predefined_outcomes(1);
predefined_outcomes_2 = get_predefined_outcomes(2);


d{1} = ones(16,1);   % context: industry identity 
d{2} = zeros(4,1);  % research process: there are 4 processes
d{2}(1) = 1;         % start from the first research process

Nf = numel(d);
for f = 1:Nf
    Ns(f) = numel(d{f});
end

% disp(AfterSim.MDP.A{2}(:, 1, :))


% Initialize new A matrices
A = AfterSim.MDP.A;
A{2} = zeros(size(AfterSim.MDP.A{2})); % Reset A{2} as it will be completely rebuilt

% Update A matrices based on new predefined outcomes
for f1 = 1:Ns(1) % context: industry identity
    for f2 = 1:Ns(2) % research process
        % Use the predefined outcome for this industry and process
        currentOutcome = predefined_outcomes_2{f1,f2};
        
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
AfterSim.MDP.eta = 0.1;
AfterSim.MDP.D = d;
AfterSim.MDP.beta = 0.5;
% if isfield(AfterSim.MDP, 's')
%     AfterSim.MDP.s = zeros(size(AfterSim.MDP.s));
% end
% if isfield(AfterSim, 's')
%     AfterSim.s = zeros(size(AfterSim.s));
% end
% AfterSim.MDP.o = zeros(size(AfterSim.MDP.o));  % Reset all observations to zero
% if isfield(AfterSim.MDP, 'O')
%     AfterSim.MDP = rmfield(AfterSim.MDP, 'O');
% end

% disp(AfterSim.MDP.A{2}(:, 1, :))


% The lowercase 'a' matrices remain unchanged
% AfterSim.MDP.a = AfterSim.MDP.a;

% Save the updated AfterSim structure
save('modified_agent_x.mat', 'AfterSim');

% Optional: Display a message to confirm the update
disp('Environment modified and AfterSim structure updated.');


N_new = 20;
% illustrate a single trial
%==========================================================================
mdp = AfterSim;
% disp(mdp.MDP.A{2}(:, 1, :))

BeforeSim = mdp; % before simulations
% Initialize array to store scores

load('scores_data.mat');

scores = [scores_1 zeros(1, N_new)]; % Extend the scores array

% Initialize a cell array to store the outputs
output_data = cell(1, N + N_new);

% Open a file to write the outputs
fileID = fopen('iteration_outputs.txt', 'w');

% Run active inference agent for N_new iterations
for i = 1:N_new
    MMDP(N + i) = spm_MDP_VB_X(mdp);
    
    % Update the a matrix with the learned probabilities
    for j = 1:numel(mdp.MDP.a)
        mdp.MDP.a{j} = MMDP(N + i).mdp(end).a{j};
    end
    
    % Capture the output of disp() into a string
    output = evalc('disp(mdp.MDP.a{2}(:, 1, :))');
    output_data{N + i} = output;
    
    % Write the output to the file
    fprintf(fileID, 'Iteration %d:\n', N + i);
    fprintf(fileID, '%s\n', output);
    fprintf(fileID, '------------------------\n');
    
    % Calculate the total score for this iteration
    score = 0;
    
    industry = 1;  % Only consider the first industry
    fprintf('Iteration %d:\n', N + i);
    for process = 1:4
        % Get the values in the lowercase 'a' matrix
        a_values = mdp.MDP.a{2}(:, industry, process);
        
        % Get the correct outcome for this process
        correct_outcome = predefined_outcomes_2{industry, process};
        
        % Map the outcome to an index
        outcome_map = containers.Map({'Excellent', 'Good', 'Neutral', 'Bad', 'Terrible'}, 1:5);
        correct_index = outcome_map(correct_outcome);
        
        % Calculate process score
        correct_belief = a_values(correct_index);
        incorrect_belief = max(a_values([1:correct_index-1, correct_index+1:end]));
        process_score = (correct_belief - incorrect_belief + 1) / 2;  % Normalize to [0, 1]
        score = score + process_score;
        
        % Debug output
        fprintf('  Process %d: Correct outcome: %s, Agent belief: %.4f, Max incorrect belief: %.4f, Process score: %.4f\n', ...
                process, correct_outcome, correct_belief, incorrect_belief, process_score);
    end
    

    scores(N + i) = score; % Store the total score for this iteration
    
    % Print the total score for this iteration
    fprintf('Iteration %d: Total Score = %.4f\n', N + i, score);
    
    % Commented out difference score calculation
    % new_score = 0;
    % if ~strcmp(predefined_outcomes_1{industry, process}, predefined_outcomes_2{industry, process})
    %     for k = 1:5
    %         if A_values(k) ~= 0
    %             new_score = new_score + (a_values(k) * A_values(k));
    %         end
    %     end
    % end
    % new_scores(N + i) = new_score; % Store the new score for this iteration
end

% Close the file
fclose(fileID);

% Save the cell array of outputs
save('iteration_outputs.mat', 'output_data');

disp('Iteration outputs have been saved to iteration_outputs.txt and iteration_outputs.mat');

% Plot the total score over time
figure;
plot(1:(N + N_new), scores, '-o');
xlabel('Iteration');
ylabel('Total Score');
title('Learning Progress of the Active Inference Agent');
grid on;
saveas(gcf, fullfile(figureFolder, 'learning_progress.png'));

% Save the cell array of outputs
save('iteration_outputs.mat', 'output_data');

disp('Iteration outputs have been saved to iteration_outputs.txt and iteration_outputs.mat');

% % Plot both scores over time including the new iterations
% figure;
% plot(1:(N + N_new), scores, 'b-', 'LineWidth', 2);
% hold on;
% plot(1:(N + N_new), new_scores, 'r-', 'LineWidth', 2);
% legend('Original Score', 'Difference Score');
% xlabel('Iteration');
% ylabel('Score');
% title('Learning Progress of the Active Inference Agent');
% grid on;
% saveas(gcf, fullfile(figureFolder, 'learning_progress_with_difference.png'));

AfterSim = mdp; % after simulations


save('all_variables_research_learner2.mat');
