% Load the previous results
load('/Users/rithvikprakki/ARC-AGI/all_variables_research_learner.mat');

figureFolder = '/Users/rithvikprakki/ARC-AGI/figures4';

% Define outcomes for each industry
predefined_outcomes_1 = get_predefined_outcomes(1);
predefined_outcomes_2 = get_predefined_outcomes(2);

d{1} = ones(16,1);   % context: industry identity 
d{2} = zeros(4,1);  % research process: there are 4 processes
d{2}(1) = 1;         % start from the first research process

Nf = numel(d);
for f = 1:Nf
    Ns(f) = numel(d{f});
end

% Number of new iterations
N_new = 50;

% Generate observations based on the differences in environments
observations = generate_differential_observations(predefined_outcomes_1, predefined_outcomes_2, N_new);

% Create a new MDP structure without the A matrix
new_mdp = struct();
new_mdp.T = AfterSim.MDP.T;  % Time steps
new_mdp.V = AfterSim.MDP.V;  % Policies
new_mdp.B = AfterSim.MDP.B;  % Transition probabilities
new_mdp.C = AfterSim.MDP.C;  % Preferences
new_mdp.D = AfterSim.MDP.D;  % Initial state probabilities
new_mdp.a = AfterSim.MDP.a;  % Concentration parameters for A (we keep this for learning)


% Set other necessary parameters
new_mdp.eta = 30;
new_mdp.beta = 0.5;

% Save the new MDP structure
save('new_mdp_without_A.mat', 'new_mdp');

disp('New MDP structure created without A matrix.');

% Initialize array to store scores
scores = [scores zeros(1, N_new)]; % Extend the scores array

% Initialize a cell array to store the outputs
output_data = cell(1, N + N_new);

% Open a file to write the outputs
fileID = fopen('iteration_outputs.txt', 'w');

MMDP = cell(1, N + N_new);

% Function to print observations with labels
function print_observation(observation, iteration)
    research_processes = {'Process 1', 'Process 2', 'Process 3', 'Process 4'};
    outcomes = {'Excellent', 'Good', 'Neutral', 'Bad', 'Terrible'};
    industries = {'Food', 'Oil', 'Gas', 'Nuclear', 'Renewable Energy', 'Automotive', 'Aviation', 'Pharmaceuticals', 'Biotechnology', 'Information Technology', 'Telecommunications', 'Finance', 'Healthcare', 'Education', 'Entertainment', 'Agriculture'};
    
    fprintf('Iteration %d:\n', iteration);
    fprintf('Research Process: %s\n', research_processes{observation(1)});
    fprintf('Outcome: %s\n', outcomes{observation(2)});
    fprintf('Industry: %s\n', industries{observation(3)});
    fprintf('------------------------\n');
end

% Open a file to write the observations
obsFileID = fopen('observation_outputs.txt', 'w');

decay_factor = 0.05;

% Initialize arrays to store scores
total_scores = zeros(1, N + N_new);
total_scores(1:N) = scores(1:N);  % Copy scores from the first run
diff_scores = zeros(1, N + N_new);  % Initialize difference scores

% Function to calculate the difference score
function diff_score = calculate_diff_score(mdp, observation, predefined_outcomes_1, predefined_outcomes_2)
    diff_score = 0;
    industry = observation(3);
    process = observation(1);
    outcome = observation(2);
    
    if ~strcmp(predefined_outcomes_1{industry, process}, predefined_outcomes_2{industry, process})
        old_outcome = find(strcmp(predefined_outcomes_1{industry, process}, {'Excellent', 'Good', 'Neutral', 'Bad', 'Terrible'}));
        new_outcome = find(strcmp(predefined_outcomes_2{industry, process}, {'Excellent', 'Good', 'Neutral', 'Bad', 'Terrible'}));
        
        old_prob = mdp.a{2}(old_outcome, industry, process);
        new_prob = mdp.a{2}(new_outcome, industry, process);
        
        if old_prob < new_prob
            diff_score = diff_score + (new_prob - old_prob);
        end
    end
end

% Run active inference agent for N_new iterations
for i = 1:N_new
    % Pass the pre-generated observations to the agent
    new_mdp.o = observations(:, i);
    
    % Print the observation
    print_observation(new_mdp.o, N + i);
    
    % Write the observation to the file
    fprintf(obsFileID, 'Iteration %d:\n', N + i);
    fprintf(obsFileID, 'Research Process: %d\n', new_mdp.o(1));
    fprintf(obsFileID, 'Outcome: %d\n', new_mdp.o(2));
    fprintf(obsFileID, 'Industry: %d\n', new_mdp.o(3));
    fprintf(obsFileID, '------------------------\n');
    
    % Store the output in the cell array
    MMDP{N + i} = spm_MDP_VB_X(new_mdp);
    
    % Update the a matrix with the learned probabilities
    for j = 1:numel(new_mdp.a)
        new_mdp.a{j} = MMDP{N + i}.a{j};
        new_mdp.a{j} = new_mdp.a{j};
        % * (1 - decay_factor);
    end
    
    % Capture the output of disp() into a string
    output = evalc('disp(new_mdp.a{2}(:, 1, :))');
    output_data{N + i} = output;
    
    % Write the output to the file
    fprintf(fileID, 'Iteration %d:\n', N + i);
    fprintf(fileID, '%s\n', output);
    fprintf(fileID, '------------------------\n');
    
    % Calculate the total score for this iteration
    total_score = calculate_score(new_mdp, observations(:, i));
    total_scores(N + i) = total_score;
    
    % Calculate the difference score for this iteration
    diff_score = calculate_diff_score(new_mdp, observations(:, i), predefined_outcomes_1, predefined_outcomes_2);
    diff_scores(N + i) = diff_score;
    
    % Print scores for this iteration
    fprintf('Iteration %d: Total Score = %.4f, Difference Score = %.4f\n', N + i, total_score, diff_score);
end

% Close the observation file
fclose(obsFileID);

% Save the cell array of outputs
save('iteration_outputs.mat', 'output_data');

disp('Iteration outputs have been saved to iteration_outputs.txt and iteration_outputs.mat');

% Plot the scores over time
figure;
hold on;

% Plot total scores
plot(1:(N + N_new), total_scores, '-o', 'DisplayName', 'Total Score');

% Plot difference scores (flat line at 0 for first N iterations)
plot([1 N], [0 0], 'r-', 'DisplayName', 'Difference Score (First Run)');
plot(N+1:(N + N_new), diff_scores(N+1:end), 'r-o', 'DisplayName', 'Difference Score (Second Run)');

xlabel('Iteration');
ylabel('Score');
title('Learning Progress of the Active Inference Agent');
legend('show');
grid on;

% Save the plot as a PNG file in the specified folder
saveas(gcf, fullfile(figureFolder, 'learning_progress_with_diff_scores.png'));

function observations = generate_differential_observations(predefined_outcomes_1, predefined_outcomes_2, num_iterations)
    % Initialize the observations matrix
    observations = zeros(3, num_iterations);
    
    % Find the differences between predefined_outcomes_1 and predefined_outcomes_2
    [diff_industries, diff_processes] = find(~cellfun(@strcmp, predefined_outcomes_1, predefined_outcomes_2));
    
    % If there are no differences, use all industries and processes
    if isempty(diff_industries)
        warning('No differences found between predefined_outcomes_1 and predefined_outcomes_2. Using all possibilities.');
        [diff_industries, diff_processes] = ndgrid(1:16, 1:4);
        diff_industries = diff_industries(:);
        diff_processes = diff_processes(:);
    end
    
    % Generate observations based on the differences
    for i = 1:num_iterations
        % Randomly select one of the cases
        idx = randi(length(diff_industries));
        industry = diff_industries(idx);
        process = diff_processes(idx);
        
        % Research Process Observation (Modality 1)
        observations(1, i) = process;
        
        % Outcome Observation (Modality 2)
        outcome = predefined_outcomes_2{industry, process};
        switch outcome
            case 'Excellent'
                obs = 1;
            case 'Good'
                obs = 2;
            case 'Neutral'
                obs = 3;
            case 'Bad'
                obs = 4;
            case 'Terrible'
                obs = 5;
            otherwise
                error('Unknown outcome: %s', outcome);
        end
        observations(2, i) = obs;
        
        % Industry Cue Observation (Modality 3)
        observations(3, i) = industry;
    end
end

% Helper function to calculate score
function score = calculate_score(mdp, observation)
    score = 0;
    for g = 1:3  % For each modality
        a_values = mdp.a{g}(:, observation(g));
        score = score + sum(a_values);
    end
end