function [ourM,problem]=copula_SDP_methodsComparison(X,M0,df)

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
       
        constant=1.5;
        X2=constant*randn(N,d);    
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
    problem.M.retr=problem.M.exp;
    
    
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

    initial_point=M0;
    
        %% Part two: compare with other methods
    color =[1 0 0; 0.4660 0.6740 0.1880;0 0.4470 0.7410;0.4940 0.1840 0.5560];
    
    arc_options.sigma_0=0.18;
    arc_options.eta_1 = 0.1;
    arc_options.eta_2 = 0.9;
    arc_options.gamma_1 = 0.7;
    arc_options.gamma_2 = 1.5;
    arc_options.gamma_3=2;
    arc_options.tolgradnorm=1e-4;
    
    [arc_M,arc_cost,arc_info, arc_options]=arc(problem,initial_point,arc_options); 
    
    
    clear FUNCTIONS
     %crc_options.sigma_0=0.12;
     crc_options.sigma_0=0.01;
     crc_options.stochastic=0;
     crc_options.tolgradnorm=1e-4;
     [crc_M,crc_cost, crc_info, crc_options]=rsvrc(problem,initial_point,crc_options);
    
    clear FUNCTIONS
    options.maxinneriter=5;
    options.sigma_0=0.01;
    options.batchsize_g=N/20;
    options.batchsize_h=N/20;
    options.tolgradnorm=1e-5;
    options.stochastic=1;
    
    [ourM,ourcost, our_info, ouroptions]=rsvrc(problem,initial_point,options);
    

    clear FUNCTIONS
    tr_options.Delta0=1;
    tr_options.tolgradnorm=1e-6;
    [tr_M,tr_cost,tr_info, tr_options]=trustregions(problem,initial_point,tr_options);
    
    
%% runtime comparison 
% %     
    runtime_comparison=figure('Name','runtime_comparison');
    A(1)=semilogy([our_info.time_cpu],[our_info.gradnorm].^(3/2),'Color',color(1,:),'LineWidth',0.2);
    hold on 
    A(2)=semilogy([tr_info.time_cpu],[tr_info.gradnorm].^(3/2),'+-','Color',color(2,:),'LineWidth',1.2);
    hold on 
    A(3)=semilogy([arc_info.time_cpu],[arc_info.gradnorm].^(3/2),'*--','Color',color(3,:),'LineWidth',1.2);
    hold on 
    A(4)=semilogy([crc_info.time_cpu],[crc_info.gradnorm].^(3/2),'o-','Color',color(4,:),'LineWidth',1.2);
      
    hold on 
    
    for rep=1:14
        clear FUNCTIONS
        options.random_seed=randi(200000);
        [ourM,ourcost, our_info, ouroptions]=rsvrc(problem,M0,options);
        semilogy([our_info.time_cpu],[our_info.gradnorm].^(3/2),'Color',color(1,:),'LineWidth',0.2);   
    end
    LegendsStrings{1}=['SVRC'];
    LegendsStrings{2}=['RTR'];
    LegendsStrings{3}=['ARC'];
    LegendsStrings{4}=['CRC'];
    
    xlabel('$cputime$','interpreter','latex','FontSize',30);
    ylabel('$\mu (\mathbf{x}^k)$','interpreter','latex','FontSize',30);
    legend(A,LegendsStrings,'interpreter','latex','FontSize',20);
    
  
    for rep=1:15
         options.random_seed=randi(200000);
         %profile on 
         [ourM,ourcost, our_info, ouroptions]=rsvrc(problem,initial_point,options);
          rep_len(rep)=size([our_info.gradnorm],2);
          rep_gradnorm(rep,1:rep_len(rep))=[our_info.gradnorm];
          rep_time(rep,1:rep_len(rep))=[our_info.time_cpu];
          rep
    end
    median_line=median(rep_gradnorm,1);
    SO_comparison=figure('Name','SO_N_comparison');
    MM=[our_info.so_count];
    A(1)=semilogy(MM(1:min(rep_len)),median_line(1:min(rep_len)).^(3/2),'LineWidth',1.0,'Color',color(1,:));
    hold on
    plot_distribution_prctile(MM(1:min(rep_len)),rep_gradnorm(:,1:min(rep_len)).^(3/2),'Prctile',[25 50 75 90],'Color',color(1,:),'LineWidth',1.0);
    hold on
    A(2)=semilogy([tr_info.iter], [tr_info.gradnorm].^(3/2), '+-','Color',color(2,:),'LineWidth',1.2);
    hold on 
    A(3)=semilogy([arc_info.iter], [arc_info.gradnorm].^(3/2), '*--','Color',color(3,:),'LineWidth',1.2);
    hold on 
    A(4)=semilogy([crc_info.so_count], [crc_info.gradnorm].^(3/2),'o-','Color',color(4,:),'LineWidth',1.2);
    LegendsStrings{1}=['SVRC'];
    LegendsStrings{2}=['RTR'];
    LegendsStrings{3}=['ARC'];
    LegendsStrings{4}=['CRC'];

    xlabel('$SO/N$','interpreter','latex','FontSize',30);
    ylabel('$\mu (\mathbf{x}^k)$','interpreter','latex','FontSize',30);
    legend(A,LegendsStrings,'interpreter','latex','FontSize',20,'location','southwest');
    
end