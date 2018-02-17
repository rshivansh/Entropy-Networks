function [Centers, betas, Theta] = trainRBFN(X_train, y_train, centersPerCategory, verbose)
% TRAINRBFN Builds an RBF Network from the provided training set.
%   [Centers, betas, Theta] = trainRBFN(X_train, y_train, centersPerCategory, verbose)
%    
%   There are three main steps to the training process:
%     1. Prototype selection through k-means clustering.
%     2. Calculation of beta coefficient (which controls the width of the 
%        RBF neuron activation function) for each RBF neuron.
%     3. Training of output weights for each category using gradient descent.
%
%   Parameters
%     X_train  - The training vectors, one per row
%     y_train  - The category values for the corresponding training vector.
%                Category values should be continuous starting from 1. (e.g.,
%                1, 2, 3, ...)
%     centersPerCategory - How many RBF centers to select per category. k-Means
%                          requires that you specify 'k', the number of 
%                          clusters to look for.
%     verbose  - Whether to print out messages about the training status.
%
%   Returns
%     Centers  - The prototype vectors stored in the RBF neurons.
%     betas    - The beta coefficient for each coressponding RBF neuron.
%     Theta    - The weights for the output layer. There is one row per neuron
%                and one column per output node / category.

% $Author: ChrisMcCormick $    $Date: 2014/08/18 22:00:00 $    $Revision: 1.3 $
    data = load('dataset.csv');
    X = data(:, 1:4);
    y = data(:,5);
    % Get the number of unique categories in the dataset.
    numCats = size(unique(y_train), 1);
    
    % Set 'm' to the number of data points.
    m = size(X_train, 1);
    
    % Ensure category values are non-zero and continuous.
    % This allows the index of the output node to equal its category (e.g.,
    % the first output node is category 1).
    if (any(y_train == 0) || any(y_train > numCats))
        error('Category values must be non-zero and continuous.');
    end
    
    % ================================================
    %       Select RBF Centers and Parameters
    % ================================================
    % Here I am selecting the cluster centers using k-Means clustering.
    % I've chosen to separate the data by category and cluster each 
    % category separately, though I've read that this step is often done 
    % over the full unlabeled dataset. I haven't compared the accuracy of 
    % the two approaches.
    
    if (verbose)
        disp('1. Selecting centers through k-Means.');
    end    
    
    Centers = [];
    betas = [];    
    
    % For each of the categories...
    for (c = 1 : numCats)

        if (verbose)
            fprintf('  Category %d centers...\n', c);
            if exist('OCTAVE_VERSION') fflush(stdout); end;
        end
        
        % Select the training vectors for category 'c'.
        Xc = X_train((y_train == c), :);

        % ================================
        %      Find cluster centers
        % ================================
        
        % Pick the first 'centersPerCategory' samples to use as the initial
        % centers.
        init_Centroids = Xc(1:centersPerCategory, :);
        
        % Run k-means clustering, with at most 100 iterations.
        [Centroids_c, memberships_c] = kMeans(Xc, init_Centroids, 100);    
        
        
        % Remove any empty clusters.
        toRemove = [];
        
        % For each of the centroids...
        for (i = 1 : size(Centroids_c, 1))
            % If this centroid has no members, mark it for removal.
            if (sum(memberships_c == i) == 0)        
                toRemove = [toRemove; i];
            end
        end
        
        % If there were empty clusters...
        if (~isempty(toRemove))
            % Remove the centroids of the empty clusters.
            Centroids_c(toRemove, :) = [];
            
            % Reassign the memberships (index values will have changed).
            memberships_c = findClosestCentroids(Xc, Centroids_c);
        end
        
        % ================================
        %    Compute Beta Coefficients
        % ================================
        if (verbose)
            fprintf('  Category %d betas...\n', c);
            if exist('OCTAVE_VERSION') fflush(stdout); end;
        end

        % Compute betas for all the clusters.
        betas_c = computeRBFBetas(Xc, Centroids_c, memberships_c);
        fprintf('\n seizecenters=%f',size(Centroids_c));
        % Add the centroids and their beta values to the network.
        Centers = [Centers; Centroids_c];
        betas = [betas; betas_c];
        
        %fprintf('\n seizecenters=%f',size(betas));
    end

    % Get the final number of RBF neurons.
    numRBFNeurons = size(Centers, 1);
    
    % ===================================
    %        Train Output Weights
    % ===================================

    % ==========================================================
    %       Compute RBF Activations Over The Training Set
    % ===========================================================
    if (verbose)
        disp('2. Calculate RBF neuron activations over full training set.');
    end

    % First, compute the RBF neuron activations for all training examples.

    % The X_activ matrix stores the RBF neuron activation values for each 
    % training example: one row per training example and one column per RBF
    % neuron.
    X_activ = zeros(m, numRBFNeurons);

    % For each training example...
    for (i = 1 : m)
       
        input = X_train(i, :);
       
       % Get the activation for all RBF neurons for this input.
        z = getRBFActivations(Centers, betas, input);
       
        % Store the activation values 'z' for training example 'i'.
        X_activ(i, :) = z';
    end

    % Add a column of 1s for the bias term.
    X_activ = [ones(m, 1), X_activ];

    % =============================================
    %        Learn Output Weights
    % =============================================

    if (verbose)
        disp('3. Learn output weights.');
    end
    
    
    m = size(X, 1);
    %numRBFNeurons = size(centroids, 1);
for i = 1:m
    input=X(i, :);
end
%phis = getRBFActivations(Centers, betas, input);
%phis = [1; phis];
    
%fprintf('\n x=%f',size(phis));
    
    % Add a 1 to the beginning of the activations vector for the bias term.
   
pop=size(Centers,1);
RUNS=30;
runs=0;
while(runs<RUNS)
%pop=10; % population size
var=1; % no. of design variables
maxFes=1950;
maxGen=maxFes;
mini=-30*ones(1,3);
maxi=30*ones(1,3);
[row,var]=size(mini);
%Theta=zeros(pop,3);
%for i=1:3
%Theta(:,i)=mini(i)+(maxi(i)-mini(i))*rand(pop,1);
%
%fprintf('\n random=%f',x);
%end
Theta = zeros(numRBFNeurons + 1, numCats);

    % For each category...
    for (c = 1 : numCats)

        % Make the y values binary--1 for category 'c' and 0 for all other
        % categories.
        y_c = (y_train == c);

        % Use the normal equations to solve for optimal theta.
        Theta(:, c) = pinv(X_activ' * X_activ) * X_activ' * y_c;
    end
    
ch=1;
gen=0;
f=myobj(X,Theta,y,Centers,betas,input,Centroids_c);
%fprintf('\n f=%f',f);
%fprintf('\n function=%f',f);
while(gen<maxGen)
Thetanew=updatepopulation(Theta,f);

Thetanew=trimr(mini,maxi,Thetanew);
%fprintf('\n xnew=%f',xnew);
fnew=myobj(X,Thetanew,y,Centers,betas,input,Centroids_c);

for i=1:3
if(fnew(i)<f(i))
Theta(i,:)=Thetanew(i,:);
f(i)=fnew(i);
end
end
%disp('%%%%%%%% Final population %%%%%%%%%');


fnew=[];Thetanew=[];
gen=gen+1;
%fopt(gen)=min(f);

end

runs=runs+1;

fopt(gen)=min(f);
[val,ind]=min(f);
Fes=pop*ind;

best(runs)=val;
end
bbest=min(best);
mbest=mean(best);
wbest=max(best);
stdbest=std(best);
mFes=mean(Fes);
stdFes=std(Fes);



%end
%fprintf('\n betas=%f',betas);
%fprintf('\n final f=%f',f);
%fprintf('\n betas=%f',10000000*betas);

%fprintf('\n betas=%f',betas);
%fprintf('\n mean=%f',mbest);
%fprintf('\n worst=%f',wbest);
%fprintf('\n std. dev.=%f',stdbest);
%fprintf('\n mean function evaluations=%f',mFes);
end
function[z]=trimr(mini,maxi,Theta)
[row,col]=size(Theta);
for i=1:col
Theta(Theta(:,i)<mini(i),i)=mini(i);
Theta(Theta(:,i)>maxi(i),i)=maxi(i);
end
z=Theta;
end
function [Thetanew]=updatepopulation(Theta,f)
[row,col]=size(Theta);

[t,tindex]=min(f);
Best=Theta(tindex,:);
[w,windex]=max(f);
worst=Theta(windex,:);
Thetanew=zeros(row,col);
for i=1:row
for j=1:col
r=rand(1,2);
Thetanew(i,j)=Theta(i,j)+r(1)*(Best(j)-abs(Theta(i,j)))-r(2)*(worst(j)-abs(Theta(i,j)));
end
end
end

function [f]=myobj(X,Theta,y,Centers,betas,input,centroids)

[r,c]=size(Theta);
%for i=1:r

%for j=1:c
numRBFNeurons = size(centroids, 1);
 %fprintf('\n insidex=%f',x);
    % Compute sigma for each cluster.
    
   % for (i = 1 : numRBFNeurons)
    % For each cluster...
    %for (u = 1 : numRBFNeurons)
        % Select the next cluster centroid.
        
%        center = centroids( u,:);
        
       % fprintf('\n center=%f',input(1));
        %fprintf('\n loopx=%f',x);
        
       
        % Select all of the members of this cluster.
        %members = X((memberships == u), :);

        % Compute the average L2 distance to all of the members. 
         % Compute the activations for each RBF neuron for this input.
        
    
    % Add a 1 to the beginning of the activations vector for the bias term.
      %  numCats = size(unique(y_train), 1);
       % for (c = 1 : numCats)

        % Make the y values binary--1 for category 'c' and 0 for all other
        % categories.
        %y_c = (y_train == c);
    %    end
     %   m = size(X, 1);
      %  for i = 1:m
       %     Y_c=y_c(i);
      %  end
       % Phis=repmat(phis,1,3);
         
          
        %u=Phis.*Theta;
       %[maxu, category] = max(u,[],2);
     %   [p,pindex]=input(1);
      %  Y_c=y_c(pindex);
      
       % fprintf('\n u=%f',size(u));
      %  fprintf('\n y_train=%f',size(y_c));
        % Subtract the center vector from each of the member vectors.
       % differences = bsxfun(@minus,u ,y_c );
        %fprintf('\n differences=%f',differences);
        phis = getRBFActivations(Centers, betas, input);
        phis = [1; phis];
        
    u = Theta' * phis;
m = size(X, 1);
for (i = 1 : m)
    % Compute the scores for both categories.
    
    
	[maxScore, category] = max(u);
	
    % Validate the result.
 %   if (category == y(i))
  %      numRight = numRight + 1;
  %  else
   %     wrong = [wrong; X(i, :)];
   % end
    


         
        % Select the next cluster centroid.
        %fprintf('\n ix=%f',x);
        %center = centroids( u,:);
        
        %fprintf('\n center=%f',center);
        %fprintf('\n loopx=%f',x);
        
       
        % Select all of the members of this cluster.
        %members = X((memberships == u), :);

        % Compute the average L2 distance to all of the members. 
        
            
        % Subtract the center vector from each of the member vectors.
       
        differences = bsxfun(@minus,category , y(i));
       % fprintf('\n differences=%f',differences);
        %fprintf('\n xinside=%f',x);
        %fprintf('\n center=%f',center);
        
        % Take the sum of the squared differences.
        sqrdDiffs = (differences .^ 2);
        
        % Take the square root to get the L2 (Euclidean) distance.
        distances = (sqrdDiffs);
        
        % Compute the average L2 distance, and use this as sigma.
        x=distances;
        y=y+x;
        z=y;
        f=z;

        
        
       end 
%end
        
%    end

        %if (any(y == 0))
         %   error('One of the sigma values is zero!');
        %end
        %betas = 1 ./ (2 .* (f.^2) );
        
        
%end
        
   end



%end


%end
% Create a matrix to hold all of the output weights.
    % There is one column per category / output neuron.
   
    
   
    



