function [ourM,problem]=sphere_methodsComparison(X,M0)

    %% Data simulation 
    rng(4);
    % If no inputs are provided, since this is an example file, simulate 
    % the data set: n samples in R^d. This is for illustration purposes only.
   
    if ~exist('X','var')||isempty(X)
        d=20;
        N=1e5;
        fprintf('Generating data...');  
        % randomly generate the true beta. 
        beta=spherefactory(d,1).rand();
        X=rand(d,N);
       
        constant=0.1;
        noise=constant*randn(1,N);
        
        flag=beta'*X+noise;
        Y=2*(flag>0)-1;   
        fprintf(' done (size: %d x %d) .\n', size(X));
    end
    [d,N]=size(X);
    assert(isreal(X),'X must be real.');
    
    if ~exist('inv_sig_0','var')|| isempty(M0)
        M0=spherefactory(d,1).rand();
    end
    
    %% Manifold setting
    manifold=spherefactory(d, 1);
    problem.M=manifold;
    
    problem.M.retr=problem.M.exp; % for fair comparison, enforcing exponential mapping.
    problem.ncostterms = N; % This parameter is set for stochastic algorithms.
    
    problem.dim=d;
    problem.M.paralleltransp=problem.M.isotransp;

    problem.cost = @cost;
    function F = cost (M)
        F=0;
        for i=1:N
            a=1/(1+exp(-Y(i)*M'*X(:,i)));
            F=F+(1-a)^2;
        end
        F=F/N;     
    end

    problem.partialegrad = @partialegrad;
    function G = partialegrad (M, sample)
        
        Xsample= X(:,sample); 
        Ysample=Y(sample);
        sample_size=size(sample,2);
        G=zeros(d,1);
        for i=1:sample_size
            t0=exp(-Ysample(i)*Xsample(:,i)'*M);
            t1=1+t0;
            G=G-(2*t0*Ysample(i)*(1-1/t1))/(t1^2)*Xsample(:,i);
        end
        G=G/N;  
        
    end

    problem.partialehess_tensor=@partialehess_tensor;
    function H_tensor=partialehess_tensor(M,sample)
        Xsample=X(:,sample);
        Ysample=Y(sample);
        sample_size=size(sample,2);
        H=zeros(d,d); 
        for i=1:sample_size
            t0=M'*Xsample(:,i);
            t1=exp(-t0*Ysample(i));
            t2=Ysample(i)^2;
            t3=exp(-2*t0*Ysample(i));
            t4=1+t1;
            T5=Xsample(:,i)*Xsample(:,i)';
            H=H-((6*t1*t2*t3)/(t4^4)*T5-(4*t2*t3)/(t4^3)*T5);
        end
        H_tensor=H/N;   
    end

    problem.tensor_factory=@tensor_factory;
    function result=tensor_factory(H,eta)
        result=H*eta;
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
        eh=store.ehess_tensor*eta;
    end
    %checkgradient(problem); 
    %checkhessian(problem);

    %% Part two: compare with other methods
    color =[1 0 0; 0.4660 0.6740 0.1880;0 0.4470 0.7410;0.4940 0.1840 0.5560];
    
    
    options.maxinneriter=5;
    options.sigma_0=0.1;
    options.batchsize_g=N/20;
    options.batchsize_h=N/20;
    options.tolgradnorm=1e-6;
    options.stochastic=1;
    
    [ourM,ourcost, our_info, ouroptions]=rsvrc(problem,M0,options);
    
     %crc_options.sigma_0=0.12;
     crc_options.sigma_0=0.1;
     crc_options.stochastic=0;
     [crc_M,crc_cost, crc_info, crc_options]=rsvrc(problem,M0,crc_options);

    tr_options.Delta0=1.5;
    [tr_M,tr_cost,tr_info, tr_options]=trustregions(problem,M0,tr_options);
    
    arc_options.sigma_0=0.1;
    arc_options.eta_1 = 0;
    arc_options.eta_2 = 0.85;
    arc_options.gamma_1 = 0.1;
    arc_options.gamma_2 = 2.5;
    
    [arc_M,arc_cost,arc_info, arc_options]=arc(problem,M0,arc_options); 
    
    
     %% runtime comparison 
    
    runtime_comparison=figure('Name','runtime_comparison');
    A(1)=semilogy([our_info.time_cpu],[our_info.gradnorm].^(3/2),'Color',color(1,:),'LineWidth',0.8);
    hold on 
    A(2)=semilogy([tr_info.time_cpu],[tr_info.gradnorm].^(3/2),'+-','Color',color(2,:),'LineWidth',1.2);
    hold on 
    A(3)=semilogy([arc_info.time_cpu],[arc_info.gradnorm].^(3/2),'*--','Color',color(3,:),'LineWidth',1.2);
    hold on 
    A(4)=semilogy([crc_info.time_cpu],[crc_info.gradnorm].^(3/2),'o-','Color',color(4,:),'LineWidth',1.2);
    
    hold on 
    
    for rep=1:14
        clear FUNCTIONS
        options.random_seed=randi(2000);
        [ourM,ourcost, our_info, ouroptions]=rsvrc(problem,M0,options);
        semilogy([our_info.time_cpu],[our_info.gradnorm].^(3/2),'Color',color(1,:),'LineWidth',0.2);   
    end
    
    LegendsStrings{1}=['SVRC'];
    LegendsStrings{2}=['RTR'];
    LegendsStrings{3}=['ARC'];
    LegendsStrings{4}=['CRC'];
    
    xlabel('$cputime$','interpreter','latex','FontSize',30);
    ylabel('$\mu (\mathbf{x}^k)$','interpreter','latex','FontSize',30);
    %ylabel('$||grad F(\mathbf{x})||$','interpreter','latex');
    legend(A,LegendsStrings,'interpreter','latex','FontSize',20);
    
    
% %% number of second order oracle comparision 
%   
%     for rep=1:15
%          options.random_seed=randi(2000);
%          clear FUNCTIONS
%          [ourM,ourcost, our_info, ouroptions]=rsvrc(problem,M0,options);
%           rep_len(rep)=size([our_info.gradnorm],2);
%           rep_gradnorm(rep,1:rep_len(rep))=[our_info.gradnorm];
%           rep_time(rep,1:rep_len(rep))=[our_info.time_cpu];
%           rep
%     end
%     median_line=median(rep_gradnorm,1);
%     SO_comparison=figure('Name','SO_N_comparison');
%     MM=[our_info.so_count];
%     A(1)=semilogy(MM(1:min(rep_len)),median_line(1:min(rep_len)).^(3/2),'LineWidth',1.0,'Color',color(1,:));
%     hold on
%     plot_distribution_prctile(MM(1:min(rep_len)),rep_gradnorm(:,1:min(rep_len)).^(3/2),'Prctile',[25 50 75 90],'Color',color(1,:),'LineWidth',1.0);
%     hold on
%     A(2)=semilogy([tr_info.iter], [tr_info.gradnorm].^(3/2), '+-','Color',color(2,:),'LineWidth',1.2);
%     hold on 
%     A(3)=semilogy([arc_info.iter], [arc_info.gradnorm].^(3/2), '*--','Color',color(3,:),'LineWidth',1.2);
%     hold on 
%     A(4)=semilogy([crc_info.so_count], [crc_info.gradnorm].^(3/2),'o-','Color',color(4,:),'LineWidth',1.2);
%     LegendsStrings{1}=['SVRC'];
%     LegendsStrings{2}=['RTR'];
%     LegendsStrings{3}=['ARC'];
%     LegendsStrings{4}=['CRC'];
% 
%     xlabel('$\frac{SO}{N}$','interpreter','latex','FontSize',30);
%     ylabel('$\mu (\mathbf{x}^k)$','interpreter','latex','FontSize',30);
%     legend(A,LegendsStrings,'interpreter','latex','FontSize',20);
   

end