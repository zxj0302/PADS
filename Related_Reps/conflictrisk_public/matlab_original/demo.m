load('C:\\Users\\27491\\\Desktop\\Research\\ECD\\PADS\\Related_Reps\\conflictrisk-public\\A.mat'); % It can be any undirected positive-weighted network with 0<=w_ij<=1
% cvx_precision best

% figure,plot(graph(A));
figure,imagesc(A),colorbar;
iter = 1;
k = 2000; % k for k/2 edges
stepsz = 1;
dim = 10;
min_eig = 1e-6;

[OptA, acr, wcr, conflicts] = WCROpt(A,iter,k,dim, min_eig);

% output the number of elements in OptA-A that is close to 1
num_close_to_one = sum(sum(abs(OptA - A) > 0.999));
disp(['Number of elements in OptA-A that are close to 1: ', num2str(num_close_to_one)]);

% draw the changes of acr and wcr
figure,plot(acr,'r','LineWidth',2);
hold on,plot(wcr,'b','LineWidth',2);
xlabel('Iteration','FontSize',20);
ylabel('Risk','FontSize',20);
legend('ACR','WCR','FontSize',20);
title('The changes of ACR and WCR','FontSize',20)
