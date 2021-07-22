function [ourM,problem]=copula_SDP_methodsOther(X,M0,df)

    %% Data simulation 
    rng(4);
    if ~exist('df','var')|| isempty(df)
        df=3;
    end
    assert(df>=1 && df==round(df),'df must be integer >=1.');  
    % If no inputs are provided, since this is an example file, simulate 
    % the data set: n samples in R^d. This is for illustration purposes only.
   
    if ~exist('X','var')||isempty(X)
        d=10;
        N=1e4;
        fprintf('Generating data...');  
        
        % randomly generate the true covariance matrix. 
        D = diag(1+100*rand(d, 1));
        [Q, ~] = qr(randn(d));
        Sigma1 = Q*D*Q'; 
        Sigma_corr=corrcov(Sigma1);
        
        X1=mvtrnd(Sigma_corr,df,N);
       
        X=X1;
        
        fprintf(' done (size: %d x %d) .\n', size(X));
    end
    [N,d]=size(X);
    assert(isreal(X),'X must be real.');
    
    if ~exist('inv_sig_0','var')|| isempty(M0)
        M0=sympositivedefinitefactory(d).rand(); % initial_solution
    end
    
    %% Manifold setting
    manifold=sympositivedefinitefactory(d);
    problem.M=manifold;
    problem.ncostterms = N; % This parameter is set for stochastic algorithms.
    
    problem.dim=d;
   
    problem.vec=@vec;
    function result=vec(M)
        result=reshape(M,[],1);
    end

    problem.mat=@mat;
    function result=mat(v)
        result=reshape(v,[d,d]);
    end

    problem.cost = @cost;
    function F = cost (M)
        F=0;
        for i=1:N
            F=F+log(1+(X(i,:)*M*X(i,:)')/df);
        end
        F=F*(d+df)/(2*N)-(1/2)*log(det(M));
    end

    problem.partialegrad = @partialegrad;
    function G = partialegrad (M, sample)   
        Xsample= X(sample, :); 
        sample_size=size(sample,2);
        G=zeros(d);
        for i=1:sample_size
            G=G+(Xsample(i,:)'*Xsample(i,:))/( df+Xsample(i,:)*M*Xsample(i,:)');
        end
        G=G*(d+df)/(2*N)-(1/2)*(sample_size/N)*eye(d)/M;
    end

    problem.partialehess_tensor=@partialehess_tensor;
    function H_tensor=partialehess_tensor(M,sample)
        Xsample=X(sample,:);
        sample_size=size(sample,2);
        H=zeros(d^2,d^2); 
        for i=1:sample_size
            XX=Xsample(i,:)'*Xsample(i,:);
            H=H+kron(XX,XX)/(df+Xsample(i,:)*M*Xsample(i,:)')^2;
        end
        M_inv=inv(M);
        H_tensor=(-(d+df)/2)*H+(1/2)*kron(M_inv,M_inv)*sample_size;
        H_tensor=H_tensor/N;
    end

    problem.tensor_factory=@tensor_factory;
    function result=tensor_factory(H,eta)
        result=mat(H*vec(eta));
    end
    
    problem.prepare=@prepare;
    function store=prepare(M,store)
        if ~isfield(store,'ehess_tensor')
            ehess_tensor=partialehess_tensor(M,1:N);
            store.ehess_tensor=ehess_tensor;
        end
    end

    problem.ehess=@ehess;
    function [eh,store]=ehess(M,eta,store)
        store=prepare(M,store);
        eh=mat(store.ehess_tensor*vec(eta));
    end


%     checkgradient(problem); 
%     checkhessian(problem);

    %% Different parameter setting
    color =[1 0 0; 0.4660 0.6740 0.1880;0 0.4470 0.7410;0.4940 0.1840 0.5560;1 1 0];
    
    marker=['o','+','*','x','>'];
    default.maxinneriter=5;
    default.sigma_0=0.01;
    default.batchsize_g=N/20;
    default.batchsize_h=N/20;
    default.tolgradnorm=1e-7;
    default.stochastic=1;
     
   
    %% Different noise_level: report the last iterate
%     sigmafig=figure('Name','Different_noise_level');
%     options=default;
%     options.tolgradnorm=1e-8;
%     noise_level=[0.1,1,5,10];
%     
%     for line=1: 4
%        constant=noise_level(line);
%        noise=constant*randn(N,d);
%        X=X1+noise;
%        
%        for dd=1:d
%            S(dd)=snr(X1(:,dd),noise(:,dd));
%        end
%          
%        SNR=mean(S)
%        
%        rep_num=15;
%        result_rep=zeros(rep_num,100);
%        rep_len=zeros(1,rep_num);
%        for rep=1:rep_num
%             options.random_seed=randi(2000);
%             [ourM,ourcost, our_info, ouroptions]=rsvrc(problem,M0,options);
%             rep_len(rep)=size([our_info.gradnorm],2);
%             result_rep(rep,1:rep_len(rep))=[our_info.gradnorm].^(3/2);
%        end
%        median_line=median(result_rep,1);
%        A(line)=semilogy(0:min(rep_len)-1,median_line(1:min(rep_len)),'LineWidth',1.0,'Color',color(line,:),'Marker',marker(line));
%        hold on
%        plot_distribution_prctile(0:min(rep_len)-1,result_rep(:,1:min(rep_len)),'Prctile',[25 50 75 90],'Color',color(line,:),'LineWidth',1.0);
%        LegendsStrings{line}= [num2str(SNR,'%.1f'),' dB'];
%        hold on 
%    end
%    xlabel('k','interpreter','latex','FontSize',20);
%    ylabel('$\mu (\mathbf{x}^k)$','interpreter','latex','FontSize',20);
%    legend(A,LegendsStrings,'interpreter','latex','FontSize',15);
   
   
   
    %% Different noise_level_average: report the averaged iterates
    sigmafig=figure('Name','Different_noise_level');
    options=default;
    options.tolgradnorm=1e-8;
    noise_level=[0.1,1,5,10];
    
    for line=1: 4
       constant=noise_level(line);
       noise=constant*randn(N,d);
       X=X1+noise;
       
       for dd=1:d
           S(dd)=snr(X1(:,dd),noise(:,dd));
       end
         
       SNR=mean(S)
       
       rep_num=15;
       result_rep=zeros(rep_num,100);
       rep_len=zeros(1,rep_num);
       for rep=1:rep_num
            options.random_seed=randi(2000);
            [ourM,ourcost, our_info, ouroptions]=rsvrc(problem,M0,options);
            rep_len(rep)=size([our_info.gradnorm],2);
            result_rep1(rep,1:rep_len(rep))=[our_info.gradnorm].^(3/2);
            for mm=2:rep_len(rep)
                result_rep(rep,mm)=mean(result_rep1(rep,2:mm));
            end
       end
       median_line=median(result_rep,1);
       A(line)=semilogy(1:min(rep_len)-1,median_line(2:min(rep_len)),'LineWidth',1.0,'Color',color(line,:),'Marker',marker(line));
       hold on
       plot_distribution_prctile(1:min(rep_len)-1,result_rep(:,2:min(rep_len)),'Prctile',[25 50 75 90],'Color',color(line,:),'LineWidth',1.0);
       LegendsStrings{line}= [num2str(SNR,'%.1f'),' dB'];
       hold on 
   end
   xlabel('k','interpreter','latex','FontSize',20);
   ylabel('$\sum\limits_{i=1}^k\mu (\mathbf{x}^i)$','interpreter','latex','FontSize',20);
   legend(A,LegendsStrings,'interpreter','latex','FontSize',15);

end