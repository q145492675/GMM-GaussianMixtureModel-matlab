%===========================================================================================
%=========================        Load Data           ======================================
clear;clc;
temp=importdata('.\Database.dat');
x=temp';
%===========================================================================================
%=========================        Initialize Data    ======================================
K=4;          
[N,M]=size(temp); 
Dimention=2;                
History=[];
iteration = 10;
%Initialize the parameter
Mean=[14 10 -4 -4; 6 -1 6 -1];
Weight = zeros(K, 1);
Covariance_Matrix = zeros(Dimention, Dimention, K);
for i = 1:K
    Weight(i) = 1 / K;
    Covariance_Matrix(:,:,i)=[1 0;0 1];
end

%=++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%==========================   The EM algorithm    ==========================================
%=++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for t=1:iteration %number of iteration
%===========================================================================================
%===============   Calculate PDF of Gaussian distribution   ================================
    Estimate_temp=zeros(K,N);
    for k=1:K
        for n=1:N
            Estimate_temp(k,n)=exp(-1/2*(x(:,n)-Mean(:,k))'*inv(Covariance_Matrix(:,:,k))*(x(:,n)-Mean(:,k)));
        end
        Estimate_temp(k,:)=Estimate_temp(k,:)/sqrt(det(Covariance_Matrix(:,:,k)));
    end
    Estimate_temp=Estimate_temp*(2*pi)^(-1/2);
%===========================================================================================
%=========================   The Estimatate Step     =======================================
    r=zeros(K,N);
    pikn=zeros(N,K);
    for n=1:N
        for k=1:K
            pikn(n,k)=Weight(k)*Estimate_temp(k,n);
            r(k,n)=Weight(k)*Estimate_temp(k,n);
        end
        r(:,n)=r(:,n)/sum(r(:,n));
    end
%===========================================================================================
%=========================   The Maximate Step    ==========================================
    Nk=zeros(1,K);
    Nk=sum(r');
    Weight=Nk/N;        % calculate Weight of Each Gaussian distribution 
    
    for k=1:K           % calculate means of Each Gaussian distribution 
        mu_k_sum=0;
        for n=1:N
            mu_k_sum=mu_k_sum+r(k,n)*x(:,n);
        end
        Mean(:,k)=mu_k_sum/Nk(k);
    end
    for k=1:K         % calculate covariance matrix of Each Gaussian distribution 
        cov_k_sum=0;
        for n=1:N
            cov_k_sum=cov_k_sum+r(k,n)*(x(:,n)-Mean(:,k))*(x(:,n)-Mean(:,k))';
        end
        Covariance_Matrix(:,:,k)=cov_k_sum/Nk(k);
    end
%===========================================================================================
%================= The Covariance of different structure of GMM  ===========================    
    for i=1:K
           Dign_Cov(:,:,i)=diag(diag(Covariance_Matrix(:,:,i)));
    end
    for i=1:K
            Avg_value(i)=sum(diag(Dign_Cov(:,:,i)))/2;
    end
    for i=1:K
            Sph_Cov(:,:,i)=eye(2,2).*Avg_value(i);
    end
    %Covariance_Matrix=Dign_Cov;            %The Dignal-Covariance matrix
    Covariance_Matrix=Sph_Cov;             %The Spherial-Covariance matrix
    %Covariance_Matrix=Covariance_Matrix     %The Full-Covariance matrix
    MLE=sum(log(sum(pikn')));
    History=[History,MLE];
end


A=1:iteration;
%figure(1);
%plot(A,History) 
%title('The Estimation of Max Likehood in each interation')
%xlabel('Iteration number')
%ylabel('The Value of MLE')
%grid on
MLE
%===========================================================================================
%=========================   The BIC of this GMM  ==========================================
km=K*Dimention+K*1/2*Dimention*(Dimention+1)+K-1;
BIC=-2*MLE+km*log(N)

%===========================================================================================
%=========================  Print the result after EM   ====================================
figure(2),hold on
for i=1:N
    [max_temp,ind_cluster]=max(r(:,i));
    if ind_cluster==1
        plot(x(1,i),x(2,i),'b*');
    end
    if ind_cluster==2
        plot(x(1,i),x(2,i),'go');
    end
    if ind_cluster==3
        plot(x(1,i),x(2,i),'r+');
    end  
    if ind_cluster==4
        plot(x(1,i),x(2,i),'magenta*');
    end  
end
legend('Class: One','Class: Two','Class: Three','Class:Four')