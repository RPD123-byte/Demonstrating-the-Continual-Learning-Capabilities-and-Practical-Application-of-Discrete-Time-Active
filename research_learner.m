% This is an active inference routine for industry research using a discrete
% state space MDP with predefined outcomes


N = 10; % Number of iterations (you can adjust this)

% First level L1

% 16 industries, each with 4 research processes. Each process has one of 5 possible outcomes.
% Agent explores the research processes where the industry is one of 16

d{1} = ones(16,1);   % context: industry identity 
d{2} = zeros(4,1);  % research process: there are 15 processes
d{2}(1) = 1;         % start from the first research process

Nf = numel(d);
for f = 1:Nf
    Ns(f) = numel(d{f});
end

No = [4 5 16]; % {15 processes}, {Excellent, Good, Neutral, Bad, Terrible}, {16 industry cues}
Ng = numel(No);

for g = 1:Ng
    A{g} = zeros([No(g),Ns]);
end

% Define outcomes for each industry and process
outcomes = {'Excellent', 'Good', 'Neutral', 'Bad', 'Terrible'};

% Predefined outcomes for each industry and process
industry_names = {'Food', 'Oil', 'Gas', 'Nuclear', 'Renewable Energy', 'Automotive', 'Aviation', 'Pharmaceuticals', 'Biotechnology', 'Information Technology', 'Telecommunications', 'Finance', 'Healthcare', 'Education', 'Entertainment', 'Agriculture'};

% Define research processes
research_processes = cell(1, 4);
research_processes{1} = 1;
research_processes{2} = 2;
research_processes{3} = 3;
research_processes{4} = 4;
% research_processes{5} = [1, 2];
% research_processes{6} = [1, 3];
% research_processes{7} = [1, 4];
% research_processes{8} = [2, 3];
% research_processes{9} = [2, 4];
% research_processes{10} = [3, 4];
% research_processes{11} = [1, 2, 3];
% research_processes{12} = [1, 2, 4];
% research_processes{13} = [1, 3, 4];
% research_processes{14} = [2, 3, 4];
% research_processes{15} = [1, 2, 3, 4];


% Initialize predefined_outcomes matrix
predefined_outcomes_1 = get_predefined_outcomes(1);
predefined_outcomes_2 = get_predefined_outcomes(2);


for f1 = 1:Ns(1) % context: industry identity
    for f2 = 1:Ns(2) % research process: one of 4
        % Use the predefined outcome for this industry and process
        currentOutcome = predefined_outcomes_1{f1,f2};
        
        A{1}(f2,f1,f2) = 1; % A{outcome modality}
                            % (process outcome, industry, process state)
                            % mapping from process to process regardless of industry
        
        A{2}(1,f1,f2) = strcmp(currentOutcome,'Excellent');
        A{2}(2,f1,f2) = strcmp(currentOutcome,'Good');
        A{2}(3,f1,f2) = strcmp(currentOutcome,'Neutral');
        A{2}(4,f1,f2) = strcmp(currentOutcome,'Bad');
        A{2}(5,f1,f2) = strcmp(currentOutcome,'Terrible');
    
        A{3}(f1,f1,f2) = 1; % (industry cue, industry identity, process)
                            % mapping from industry type to itself i.e. each
                            % industry has one cue
    end
end

for g = 1:Ng
    A{g} = double(A{g});
    a{g} = A{g};                                 
end

a{1} = 512*A{1};
a{2} = power(10,-1)*ones(size(A{2}));
% a{3} = power(10,-1)*ones(size(A{3}));


% return
% a{2} = 0.1 * ones(size(A{2}));
% a{2}(A{2} == 1) = 0.4;

% a{3} = power(10,-1)*ones(size(A{3}));

% controlled transitions: B{f} for each factor
%--------------------------------------------------------------------------
for f = 1:Nf
    B{f} = eye(Ns(f));
end

% B{1}: Industry identity transitions (no transitions allowed)
% This is already an identity matrix from the loop above, so no change needed

% B{2}: Research process transitions
%--------------------------------------------------------------------------
nu = Ns(2);  % number of actions = number of research processes (now 15)
B{2} = zeros(Ns(2), Ns(2), nu);
for k = 1:nu
    B{2}(:,:,k) = eye(Ns(2));  % Start with identity matrix
    B{2}(k,:,k) = 1;  % Set the k-th row to 1 for the k-th action
end

% Allowable policies (3 moves): V
%--------------------------------------------------------------------------
% nu = Ns(2);  % number of possible actions (equal to number of research processes)
% T = 3;       % number of time steps (3 steps per policy)
% num_policies = 64;  % number of policies to include
% 
% % Generate all possible policies
% all_policies = zeros(T, nu^3);
% policy_index = 1;
% for i = 1:nu
%     for j = 1:nu
%         for k = 1:nu
%             all_policies(:, policy_index) = [i; j; k];
%             policy_index = policy_index + 1;
%         end
%     end
% end
% 
% % Randomly select 100 policies
% random_indices = randperm(nu^3, num_policies);
% selected_policies = all_policies(:, random_indices);
% 
% % Initialize V with the correct structure
% V = zeros(T, num_policies, 2);  % (time steps, policy, state factor)
% 
% % Assign policies
% % For the context (industry) factor, always 1
% V(:,:,1) = 1;
% 
% % For the choice (research process) factor
% V(:,:,2) = all_policies;

% Allowable policies (3 moves): V
%--------------------------------------------------------------------------
nu = Ns(2);  % number of possible actions (equal to number of research processes)
T = 3;       % number of time steps (3 steps per policy)
num_policies = 64;  % number of policies to include

V = [];
for i = 1:nu
    for j = 1:nu
        for k = 1:nu
            V(:,end + 1,2) = [i;j;k]; 
        end
    end
end

% Set all zero values to 1 (this handles the context factor)
V(V==0) = 1;

% If you need to limit the number of policies:
if size(V,2) > num_policies
    random_indices = randperm(size(V,2), num_policies);
    V = V(:,random_indices,:);
end


% Set up C matrices for preferences
%--------------------------------------------------------------------------
T = 4;  % number of time steps

% C{1}: Preferences over research processes (slight preference for longer processes)
C{1} = zeros(4, T);
for i = 1:4
    process_length = numel(research_processes{i});
    C{1}(i,:) = 0.1 * process_length;  % Slight preference, scales with process length
end

% C{2}: Preferences over outcomes (strong preference for excellent and good)
C{2} = zeros(5, T);
C{2}(1,:) = 1;  % Excellent
C{2}(2,:) = 1;  % Good
C{2}(3,:) = 1;  % Neutral
C{2}(4,:) = 1; % Bad
C{2}(5,:) = 1; % Terrible

% C{3}: No preferences over industry cues
C{3} = zeros(16, T);

% MDP Structure
%--------------------------------------------------------------------------
mdp.T = T;                      % number of time steps
mdp.a = a;                      % 1st level likelihood conc. parameters
mdp.a0 = a;                     % observation model
mdp.A = A;                      % observation model
mdp.B = B;                      % transition probabilities
mdp.C = C;                      % prior over preferences
mdp.D = d;                      % prior over initial states
mdp.V = V;                      % allowable policies

mdp.chi = -30;                  % Occam's threshold
mdp.eta = 0.1;


mdp.Aname = {'Research Process', 'Outcome', 'Industry Cue'};
mdp.Bname = {'Industry', 'Research Process'};

% Clear unnecessary variables
clear A B D

% Check and finalize the MDP structure
MDP = spm_MDP_check(mdp);

% Clear all variables except MDP and N
clearvars -except MDP N predefined_outcomes_1 predefined_outcomes_2

% Second level L2

% The agent starts from industry 1. It moves between four other different 
% industries for every iteration

% prior beliefs about initial states (in terms of counts)
%--------------------------------------------------------------------------
d{1} = ones(16,1); % context (hidden state); 

% probabilistic mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------
Nf = numel(d); 
for f = 1:Nf 
    Ns(f) = numel(d{f}); 
end

No = [16]; % outcome modality: type of industry i.e., industry cue 

A{1} = eye(16);

% controllable fixation points: move to the k-th industry
%--------------------------------------------------------------------------
for k = 1:Ns(1) 
   B{1}(:,:,k) = zeros(Ns(1),Ns(1)); 
   B{1}(k,:,k) = ones(1,16);
end  
 
% allowable policies (here, specified as the next action) U
%--------------------------------------------------------------------------
Np = size(B{1},3); % number of actions
U  = ones(1,Np,Nf);
U(:,:,1) = 1:Np; % number of policies 
 
% priors: (utility) C
%--------------------------------------------------------------------------
T = 4; 
C{1} = zeros(No(1),T); % no prior preferences on this level

% MDP Structure
%--------------------------------------------------------------------------
mdp.MDP = MDP;
mdp.link = [1; 0]; % outcome of L2 enter at L1 as initial state for context 

% MDP Structure - this will be used to generate arrays for multiple trials
%==========================================================================
mdp.T = T;                      % number of moves
mdp.U = U;                      % allowable actions
mdp.A = A;                      % observation model
mdp.B = B;                      % transition probabilities
mdp.C = C;                      % preferred outcomes
mdp.D = d;                      % prior over initial states
mdp.s = [1]';                   % initial states 

mdp.Aname = {'Industry identity'}; 
mdp.Bname = {'Context'};

mdp = spm_MDP_check(mdp);

% illustrate a single trial
%==========================================================================

BeforeSim = mdp; % before simulations

% Initialize arrays to store scores
scores_1 = zeros(1, N);  % For the first industry
scores_all = zeros(1, N);  % For all 16 industries


% Initialize a cell array to store the outputs
output_data = cell(1, N);

% Open a file to write the outputs
fileID = fopen('iteration_outputs_all_industries.txt', 'w');



for i = 1:N
    MMDP(i) = spm_MDP_VB_X(mdp);
    % Update the a matrix with the learned probabilities
    for j = 1:numel(mdp.MDP.a)
        mdp.MDP.a{j} = MMDP(i).mdp(end).a{j};
    end

    output = '';
    for industry = 1:16
        output = [output sprintf('Industry %d:\n', industry)];
        for process = 1:4
            output = [output sprintf('Process %d:\n', process)];
            
            % Get the correct outcome for this process
            correct_outcome = predefined_outcomes_1{industry, process};
            outcome_map = containers.Map({'Excellent', 'Good', 'Neutral', 'Bad', 'Terrible'}, 1:5);
            correct_index = outcome_map(correct_outcome);
            
            % Display a values with an asterisk for the correct outcome
            a_values = mdp.MDP.a{2}(:, industry, process);
            for k = 1:5
                if k == correct_index
                    output = [output sprintf('*%.4f\n', a_values(k))];
                else
                    output = [output sprintf(' %.4f\n', a_values(k))];
                end
            end
            output = [output sprintf('\n')];
        end
        output = [output sprintf('\n')];
    end
    output_data{i} = output;
    
    % Write the output to the file
    fprintf(fileID, 'Iteration %d:\n', i);
    fprintf(fileID, '%s\n', output);
    fprintf(fileID, '------------------------\n');


    % Calculate the score for the first industry
    score_1 = 0;
    industry = 1;
    fprintf('Iteration %d (Industry 1):\n', i);
    for process = 1:4
        % Get the values in the lowercase 'a' matrix
        a_values = mdp.MDP.a{2}(:, industry, process);
        % Get the correct outcome for this process
        correct_outcome = predefined_outcomes_1{industry, process};
        % Map the outcome to an index
        outcome_map = containers.Map({'Excellent', 'Good', 'Neutral', 'Bad', 'Terrible'}, 1:5);
        correct_index = outcome_map(correct_outcome);
        % Calculate process score
        correct_belief = a_values(correct_index);
        incorrect_belief = max(a_values([1:correct_index-1, correct_index+1:end]));
        process_score = (correct_belief - incorrect_belief + 1) / 2; % Normalize to [0, 1]
        score_1 = score_1 + process_score;
        % Debug output
        fprintf('  Process %d: Correct outcome: %s, Agent belief: %.4f, Max incorrect belief: %.4f, Process score: %.4f\n', ...
                process, correct_outcome, correct_belief, incorrect_belief, process_score);
    end
    scores_1(i) = score_1;
    fprintf('  Total Score (Industry 1) = %.4f\n\n', score_1);

    % Calculate the score for all 16 industries
    score_all = 0;
    fprintf('Iteration %d:\n', i);
    for industry = 1:16
        industry_score = 0;
        for process = 1:4
            % Get the values in the lowercase 'a' matrix
            a_values = mdp.MDP.a{2}(:, industry, process);
            % Get the correct outcome for this process
            correct_outcome = predefined_outcomes_1{industry, process};
            % Map the outcome to an index
            outcome_map = containers.Map({'Excellent', 'Good', 'Neutral', 'Bad', 'Terrible'}, 1:5);
            correct_index = outcome_map(correct_outcome);
            % Calculate process score
            correct_belief = a_values(correct_index);
            incorrect_belief = max(a_values([1:correct_index-1, correct_index+1:end]));
            process_score = (correct_belief - incorrect_belief + 1) / 2; % Normalize to [0, 1]
            industry_score = industry_score + process_score;
            
            % Debug output for each process
            fprintf('  Industry %d, Process %d: Correct outcome: %s, Correct belief: %.4f, Max incorrect belief: %.4f, Process score: %.4f\n', ...
                    industry, process, correct_outcome, correct_belief, incorrect_belief, process_score);
        end
        score_all = score_all + industry_score;
        fprintf('  Industry %d Total Score: %.4f\n', industry, industry_score);
    end
    scores_all(i) = score_all;
    fprintf('Iteration %d: Total Score (All Industries) = %.4f\n\n', i, score_all);
end

fclose(fileID);

save('iteration_outputs_all_industries.mat', 'output_data');

figureFolder = '/Users/computer/ARC-AGI/figures2';


% Plot and save the scores for the first industry
figure;
plot(1:N, scores_1, '-o');
xlabel('Iteration');
ylabel('Score');
title('Learning Progress of the Active Inference Agent (Industry 1)');
grid on;
saveas(gcf, fullfile(figureFolder, 'learning_progress_industry1.png'));

% Plot and save the scores for all industries
figure;
plot(1:N, scores_all, '-o');
xlabel('Iteration');
ylabel('Score');
title('Learning Progress of the Active Inference Agent (All Industries)');
grid on;
saveas(gcf, fullfile(figureFolder, 'learning_progress_all_industries.png'));

% Save the scores for later use
save('scores_data.mat', 'scores_1', 'scores_all');


AfterSim = mdp; % after simulations

save('all_variables_research_learner.mat');
