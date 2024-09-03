% This is an active inference routine for industry research using a discrete
% state space MDP with predefined outcomes

% Clear workspace and add necessary paths (adjust as needed)
addpath('/Users/rithvikprakki/ARC-AGI/spm12');
addpath('/Users/rithvikprakki/ARC-AGI/spm12/toolbox/DEM');
addpath('/Users/rithvikprakki/ARC-AGI/matlab_scripts');

clear all

N = 1; % Number of iterations (you can adjust this)

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
a{2} = power(10,-1)*A{2};

% a{2} = 0.1 * ones(size(A{2}));
% a{2}(A{2} == 1) = 0.4;

a{3} = power(10,-1)*ones(size(A{3}));

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
C{2}(1,:) = 2;  % Excellent
C{2}(2,:) = 1;  % Good
C{2}(3,:) = 0;  % Neutral
C{2}(4,:) = 1; % Bad
C{2}(5,:) = 2; % Terrible

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
mdp.eta = 0.01;


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

% Initialize array to store scores
scores = zeros(1, N);

for i = 1:N
    MMDP(i) = spm_MDP_VB_X(mdp);
    % Update the a matrix with the learned probabilities
    for j = 1:numel(mdp.MDP.a)
        mdp.MDP.a{j} = MMDP(i).mdp(end).a{j};
    end
    
    % Calculate the score for this iteration
    score = 0;
    
    for industry = 1:16
        for process = 1:4
            % Get the values in the lowercase 'a' matrix
            a_values = mdp.MDP.a{2}(:, industry, process);
            % Get the values in the uppercase 'A' matrix
            A_values = mdp.MDP.A{2}(:, industry, process);
            
            % Find the index of the true result (1 in the A matrix)
            true_result_index = find(A_values == 1);
            
            % Add the corresponding value from the 'a' matrix to the score
            score = score + a_values(true_result_index);
        end
    end
    
    scores(i) = score; % Store the score for this iteration
    
    % Print the total score for this iteration
    fprintf('Iteration %d: Total Score = %.4f\n', i, score);
end

figureFolder = '/Users/rithvikprakki/ARC-AGI/figures3';


% Plot the score over time
figure;
plot(1:N, scores, '-o');
xlabel('Iteration');
ylabel('Score');
title('Learning Progress of the Active Inference Agent');
grid on;

% Save the plot as a PNG file in the specified folder
saveas(gcf, fullfile(figureFolder, 'learning_progress.png'));


AfterSim = mdp; % after simulations


% save agent_x BeforeSim AfterSim MMDP

% 
% % Display matrices/tensors for BeforeSim
% disp('Displaying and saving matrices/tensors for BeforeSim:');
% 
% % Level 1 (MDP)
% disp('Level 1 (MDP):');
% for i = 1:length(BeforeSim.MDP.a)
%     display_matrix(BeforeSim.MDP.a{i}, sprintf('BeforeSim_Level1_littlea%d', i), figureFolder);
% end
% for i = 1:length(BeforeSim.MDP.A)
%     display_matrix(BeforeSim.MDP.A{i}, sprintf('BeforeSim_Level1_bigA%d', i), figureFolder);
% end
% % for i = 1:length(BeforeSim.MDP.B)
% %     display_matrix(BeforeSim.MDP.B{i}, sprintf('BeforeSim_Level1_B%d', i), figureFolder);
% % end
% % for i = 1:length(BeforeSim.MDP.C)
% %     display_matrix(BeforeSim.MDP.C{i}, sprintf('BeforeSim_Level1_C%d', i), figureFolder);
% % end
% % display_matrix(BeforeSim.MDP.D, 'BeforeSim_Level1_D', figureFolder);
% % display_matrix(BeforeSim.MDP.V, 'BeforeSim_Level1_V', figureFolder);
% 
% % Level 2
% disp('Level 2:');
% for i = 1:length(BeforeSim.A)
%     display_matrix(BeforeSim.A{i}, sprintf('BeforeSim_Level2_A%d', i), figureFolder);
% end
% % for i = 1:length(BeforeSim.B)
% %     display_matrix(BeforeSim.B{i}, sprintf('BeforeSim_Level2_B%d', i), figureFolder);
% % end
% % for i = 1:length(BeforeSim.C)
% %     display_matrix(BeforeSim.C{i}, sprintf('BeforeSim_Level2_C%d', i), figureFolder);
% % end
% % display_matrix(BeforeSim.D, 'BeforeSim_Level2_D', figureFolder);
% % display_matrix(BeforeSim.U, 'BeforeSim_Level2_U', figureFolder);
% 
% % Display matrices/tensors for AfterSim
% disp('Displaying and saving matrices/tensors for AfterSim:');
% 
% % Level 1 (MDP)
% disp('Level 1 (MDP):');
% for i = 1:length(AfterSim.MDP.a)
%     display_matrix(AfterSim.MDP.a{i}, sprintf('AfterSim_Level1_littlea%d', i), figureFolder);
% end
% % for i = 1:length(AfterSim.MDP.A)
% %     display_matrix(AfterSim.MDP.A{i}, sprintf('AfterSim_Level1_bigA%d', i), figureFolder);
% % end
% % for i = 1:length(AfterSim.MDP.B)
% %     display_matrix(AfterSim.MDP.B{i}, sprintf('AfterSim_Level1_B%d', i), figureFolder);
% % end
% % for i = 1:length(AfterSim.MDP.C)
% %     display_matrix(AfterSim.MDP.C{i}, sprintf('AfterSim_Level1_C%d', i), figureFolder);
% % end
% % display_matrix(AfterSim.MDP.D, 'AfterSim_Level1_D', figureFolder);
% % display_matrix(AfterSim.MDP.V, 'AfterSim_Level1_V', figureFolder);
% 
% % Level 2
% disp('Level 2:');
% for i = 1:length(AfterSim.A)
%     display_matrix(AfterSim.A{i}, sprintf('AfterSim_Level2_A%d', i), figureFolder);
% end
% % for i = 1:length(AfterSim.B)
% %     display_matrix(AfterSim.B{i}, sprintf('AfterSim_Level2_B%d', i), figureFolder);
% % end
% % for i = 1:length(AfterSim.C)
% %     display_matrix(AfterSim.C{i}, sprintf('AfterSim_Level2_C%d', i), figureFolder);
% % end
% % display_matrix(AfterSim.D, 'AfterSim_Level2_D', figureFolder);
% % display_matrix(AfterSim.U, 'AfterSim_Level2_U', figureFolder);
% 
% disp('All matrices/tensors have been displayed and saved.');
% 
save('/Users/rithvikprakki/ARC-AGI/all_variables_research_learner.mat');
