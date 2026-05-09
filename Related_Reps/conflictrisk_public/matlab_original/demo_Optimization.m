load('C:\\Users\\27491\\\Desktop\\Research\\ECD\\PADS\\Related_Reps\\conflictrisk-public\\adjacency_matrix.mat'); % It can be any undirected positive-weighted network with 0<=w_ij<=1
% cvx_precision high

% figure,plot(graph(A));
figure,imagesc(A),colorbar;
%%
% options for the four conflict measures:
m = 3; % [1 for the internal conflict - ic; 
%         2 for the external conflict - ec; 
%         3 for the controversy - c; 
%         4 for the resistance - r]

% options for the two methods
gradient = 1; % [1 for projected gradient descent; 
%                0 for coordinate descent]

% options for both case internal opinions
avgCase = 0; % [1 for average case s; 
%               0 for the worst case s]

iter = 1;
k = 2000; % k for k/2 edges
stepsz = 1;
dim = 10;
min_eig = 1e-6;

[OptA, acr, wcr, conflicts] = ConflictRiskOptimization(A,m,gradient,avgCase,iter,k,stepsz,dim, min_eig);

% save the results
% save('~/Desktop/PADS/Related_Reps/conflictrisk-public/OptA.mat','OptA');
% save('~/Desktop/PADS/Related_Reps/conflictrisk-public/acr.mat','acr');
% save('~/Desktop/PADS/Related_Reps/conflictrisk-public/wcr.mat','wcr');
% save('~/Desktop/PADS/Related_Reps/conflictrisk-public/conflicts.mat','conflicts');

% draw the changes of acr and wcr
figure,plot(acr,'r','LineWidth',2);
hold on,plot(wcr,'b','LineWidth',2);
xlabel('Iteration','FontSize',20);
ylabel('Risk','FontSize',20);
legend('ACR','WCR','FontSize',20);
title('The changes of ACR and WCR','FontSize',20);

