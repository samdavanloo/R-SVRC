function [ourM,problem]=copula_SDP_methodsSimulation(X,M0,df)

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
       
        constant=15;
        X2=constant*randn(N,d);
        
        X3=zeros(N,d); 
        for i=1:N
            if mod(i,2)==0 
                X3(i,:)= sqrt(i) * randn(1,d);
            else 
                X3(i,:)=sqrt(i)*rand(1,d);
            end
        end
        
        X=X1+X2;
        fprintf(' done (size: %d x %d) .\n', size(X));
    end
    [N,d]=size(X);
    assert(isreal(X),'X must be real.');
    
    if ~exist('inv_sig_0','var')|| isempty(M0)
        M0=sympositivedefinitefactory(d).rand(); 
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
    color =[1 0 0; 0.4660 0.6740 0.1880;0 0.4470 0.7410;0.4940 0.1840 0.5560];
    
    marker=['o','+','*','x'];
    default.maxinneriter=5;
    default.sigma_0=0.01;
    default.batchsize_g=N/20;
    default.batchsize_h=N/20;
    default.tolgradnorm=1e-7;
    default.stochastic=1;
   
    
%     %% Different cubic penalty
%     sigmafig=figure('Name','Different_cubic_penalty');
%     options=default;
%     options.tolgradnorm=1e-5;
%     sigma=[0.001,0.01,0.05,0.1];
%     
%     for line=1: 4
%        options.sigma_0=sigma(line);
%        rep_num=5;
%        result_rep=zeros(rep_num,100);
%        rep_len=zeros(1,rep_num);
%        for rep=1:rep_num
%             options.random_seed=randi(20000);
%             [ourM,ourcost, our_info, ouroptions]=rsvrc(problem,M0,options);
%             rep_len(rep)=size([our_info.gradnorm],2);
%             result_rep(rep,1:rep_len(rep))=[our_info.gradnorm];
%             rep
%        end
%        median_line=median(result_rep,1);
%        A(line)=semilogy(0:min(rep_len)-1,median_line(1:min(rep_len)).^(3/2),'LineWidth',1.0,'Color',color(line,:),'Marker',marker(line));
%        hold on
%        plot_distribution_prctile(0:min(rep_len)-1,result_rep(:,1:min(rep_len)).^(3/2),'Prctile',[25 50 75 90],'Color',color(line,:),'LineWidth',1.0);
%        LegendsStrings{line}= ['$\sigma=$ ',num2str(options.sigma_0)];
%        hold on 
%    end
%    xlabel('k','interpreter','latex');
%    ylabel('$\mu (\mathbf{x}^k)$','interpreter','latex');
%    legend(A,LegendsStrings,'interpreter','latex');
%     
%     
%     %% Different T
%     Tfig=figure('Name','Different_T');
%     options=default;
%     options.tolgradnorm=1e-5;
%     T=[3,5,8,12];
%     
%     for line=1: 4
%        options.maxinneriter=T(line);
%        rep_num=15;
%        result_rep=zeros(rep_num,100);
%        rep_len=zeros(1,rep_num);
%        for rep=1:rep_num
%             options.random_seed=randi(5000);
%             [ourM,ourcost, our_info, ouroptions]=rsvrc(problem,M0,options);
%             rep_len(rep)=size([our_info.gradnorm],2);
%             result_rep(rep,1:rep_len(rep))=[our_info.gradnorm];
%        end
%        median_line=median(result_rep,1);
%        A(line)=semilogy(0:min(rep_len)-1,median_line(1:min(rep_len)).^(3/2),'LineWidth',1.0,'Color',color(line,:),'Marker',marker(line));
%        hold on
%        plot_distribution_prctile(0:min(rep_len)-1,result_rep(:,1:min(rep_len)).^(3/2),'Prctile',[25 50 75 90],'Color',color(line,:),'LineWidth',1.0);
%        LegendsStrings{line}= ['$T=$ ',num2str(options.maxinneriter)];
%        hold on 
%    end
%    xlabel('k','interpreter','latex');
%    ylabel('$\mu (\mathbf{x}^k)$','interpreter','latex');
%    legend(A,LegendsStrings,'interpreter','latex');
    
    %% Different batchsize 
    batchsizefig=figure('Name','Different_batchsize');
    options=default;
    batchsize_g=[1e2,5e2,5e1,2e3];
    batchsize_h=[1e2,5e1,5e2,2e3];
    
    for line=1: 4
       options.batchsize_g=batchsize_g(line);
       options.batchsize_h=batchsize_h(line);
       rep_num=15;
       result_rep=zeros(rep_num,100);
       rep_len=zeros(1,rep_num);
       for rep=1:rep_num
            options.random_seed=randi(200);
            [ourM,ourcost, our_info, ouroptions]=rsvrc(problem,M0,options);
            rep_len(rep)=size([our_info.gradnorm],2);
            result_rep(rep,1:rep_len(rep))=[our_info.gradnorm];
       end
       median_line=median(result_rep,1);
       A(line)=semilogy(0:min(rep_len)-1,median_line(1:min(rep_len)).^(3/2),'LineWidth',1.0,'Color',color(line,:),'Marker',marker(line));
       hold on
       plot_distribution_prctile(0:min(rep_len)-1,result_rep(:,1:min(rep_len)).^(3/2),'Prctile',[25 50 75 90],'Color',color(line,:),'LineWidth',1.0);
       LegendsStrings{line}= ['$b_g=$ ',num2str(options.batchsize_g,'%.0e'),'{ }$b_h=$ ',num2str(options.batchsize_h,'%.0e')];
       hold on 
   end
   xlabel('k','interpreter','latex');
   ylabel('$\mu (\mathbf{x}^k)$','interpreter','latex');
   legend(A,LegendsStrings,'interpreter','latex');

end