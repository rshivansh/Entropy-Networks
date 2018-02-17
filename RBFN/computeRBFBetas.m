function betas = computeRBFBetas(X, centroids, memberships)
numRBFNeurons = size(centroids, 1);
pop=numRBFNeurons;
RUNS=30;
runs=0;
while(runs<RUNS)
%pop=10; % population size
var=1; % no. of design variables
maxFes=500;
maxGen=floor(maxFes/pop);
mini=-30*ones(1,var);
maxi=30*ones(1,var);
[row,var]=size(mini);
x=zeros(pop,var);
for i=1:var
x(:,i)=mini(i)+(maxi(i)-mini(i))*rand(pop,1);
%fprintf('\n x=%f',x);
%fprintf('\n random=%f',x);
end

ch=1;
gen=0;
f=myobj(x,centroids,memberships);
%fprintf('\n f=%f',f);
%fprintf('\n function=%f',f);
while(gen<maxGen)
xnew=updatepopulation(x,f);

xnew=trimr(mini,maxi,xnew);
%fprintf('\n xnew=%f',xnew);
fnew=myobj(xnew,centroids,memberships);

%fprintf('\n fnew=%f',fnew);
for i=1:pop
if(fnew(i)<f(i))
%disp('hi')
x(i,:)=xnew(i,:);
f(i)=fnew(i);
end
end
%disp('%%%%%%%% Final population %%%%%%%%%');
%disp([x,f]);

fnew=[];xnew=[];
gen=gen+1;
fopt(gen)=min(f);

end

runs=runs+1;
[val,ind]=min(fopt);
Fes(runs)=pop*ind;
best(runs)=val;
end
bbest=min(best);
mbest=mean(best);
wbest=max(best);
stdbest=std(best);
mFes=mean(Fes);
stdFes=std(Fes);
%fprintf('\n best=%f',bbest);
betas=zeros(numRBFNeurons,1);
%for i= 1:10;
%q=f(i);  
%fprintf('\n q=%f',q);
%w=q^2;
%fprintf('\n w=%f',w);

betas = 1 ./ (2 .* (f.^2) );

%end
%fprintf('\n betas=%f',betas);
%fprintf('\n final f=%f',f);
fprintf('\n betas=%f',10000000*betas);

%fprintf('\n betas=%f',betas);
%fprintf('\n mean=%f',mbest);
%fprintf('\n worst=%f',wbest);
%fprintf('\n std. dev.=%f',stdbest);
%fprintf('\n mean function evaluations=%f',mFes);
end
function[z]=trimr(mini,maxi,x)
[row,col]=size(x);
for i=1:col
x(x(:,i)<mini(i),i)=mini(i);
x(x(:,i)>maxi(i),i)=maxi(i);
end
z=x;
end
function [xnew]=updatepopulation(x,f)
[row,col]=size(x);
[t,tindex]=min(f);
Best=x(tindex,:);
[w,windex]=max(f);
worst=x(windex,:);
xnew=zeros(row,col);
for i=1:row
for j=1:col
r=rand(1,2);
xnew(i,j)=x(i,j)+r(1)*(Best(j)-abs(x(i,j)))-r(2)*(worst(j)-abs(x(i,j)));
end
end
end
function [f]=myobj(x,centroids,memberships)

[r,c]=size(x);
%for i=1:r
y=0;
%for j=1:c
 numRBFNeurons = size(centroids, 1);
 %fprintf('\n insidex=%f',x);
    % Compute sigma for each cluster.
    
    
    % For each cluster...
    for (u = 1 : numRBFNeurons)
        % Select the next cluster centroid.
        %fprintf('\n ix=%f',x);
        center = centroids( u,:);
        
        %fprintf('\n center=%f',center);
        %fprintf('\n loopx=%f',x);
        
       
        % Select all of the members of this cluster.
        %members = X((memberships == u), :);

        % Compute the average L2 distance to all of the members. 
        
            
        % Subtract the center vector from each of the member vectors.
        differences = bsxfun(@minus,x , center);
        %fprintf('\n differences=%f',differences);
        %fprintf('\n xinside=%f',x);
        %fprintf('\n center=%f',center);
        
        % Take the sum of the squared differences.
        sqrdDiffs = sum(differences .^ 2, 2);
        
        % Take the square root to get the L2 (Euclidean) distance.
        distances = sqrt(sqrdDiffs);
        
        % Compute the average L2 distance, and use this as sigma.
        x(u, :) = mean(distances);
        y=y+x;
        z=y;
        f=z;

        if (any(y == 0))
            error('One of the sigma values is zero!');
        end
        betas = 1 ./ (2 .* (f.^2) );
        
        
%end
        
    end



end


%end
