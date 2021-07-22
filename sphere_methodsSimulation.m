function [ourM,problem]=sphere_methodsSimulation(X,M0)

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
        %M0=sympositivedefinitefactory(d).rand(); 
        M0=spherefactory(d,1).rand();
        %M0=spherefactory(d,1).rand()% initial_solution
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


    %% Different parameter setting
    color =[1 0 0; 0.4660 0.6740 0.1880;0 0.4470 0.7410;0.4940 0.1840 0.5560];
    marker=['o','+','*','x'];
    default.maxinneriter=5;
    default.sigma_0=0.1;
    default.batchsize_g=N/20;
    default.batchsize_h=N/20;
    default.tolgradnorm=1e-7;
    default.stochastic=1;
    
    
    %% Different cubic penalty
    sigmafig=figure('Name','Different_cubic_penalty');
    options=default;
    options.tolgradnorm=1e-6;
    sigma=[0.05,0.1,0.5,1];
    
    for line=1: 4
       options.sigma_0=sigma(line);
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
       LegendsStrings{line}= ['$\sigma=$ ',num2str(options.sigma_0)];
       hold on 
   end
   xlabel('k','interpreter','latex');
   ylabel('$\mu (\mathbf{x}^k)$','interpreter','latex');
   legend(A,LegendsStrings,'interpreter','latex');
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
%             options.random_seed=randi(200);
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
%     
%     %% Different batchsize 
%     batchsizefig=figure('Name','Different_batchsize');
%     options=default;
%     batchsize_g=[1e3,5e3,5e2,2e4];
%     batchsize_h=[1e3,5e2,5e3,2e4];
%     
%     for line=1: 4
%        options.batchsize_g=batchsize_g(line);
%        options.batchsize_h=batchsize_h(line);
%        if line==3
%            options.tolgradnorm=1e-6;
%        end
%        rep_num=15;
%        result_rep=zeros(rep_num,100);
%        rep_len=zeros(1,rep_num);
%        for rep=1:rep_num
%             options.random_seed=randi(200);
%             [ourM,ourcost, our_info, ouroptions]=rsvrc(problem,M0,options);
%             rep_len(rep)=size([our_info.gradnorm],2);
%             result_rep(rep,1:rep_len(rep))=[our_info.gradnorm];
%        end
%        median_line=median(result_rep,1);
%        A(line)=semilogy(0:min(rep_len)-1,median_line(1:min(rep_len)).^(3/2),'LineWidth',1.0,'Color',color(line,:),'Marker',marker(line));
%        hold on
%        plot_distribution_prctile(0:min(rep_len)-1,result_rep(:,1:min(rep_len)).^(3/2),'Prctile',[25 50 75 90],'Color',color(line,:),'LineWidth',1.0);
%        LegendsStrings{line}= ['$b_g=$ ',num2str(options.batchsize_g,'%.0e'),'{ }$b_h=$ ',num2str(options.batchsize_h,'%.0e')];
%        hold on 
%        options.tolgradnorm=1e-7;
%    end
%    xlabel('k','interpreter','latex');
%    ylabel('$\mu (\mathbf{x}^k)$','interpreter','latex');
%    legend(A,LegendsStrings,'interpreter','latex');
   
    

    
end