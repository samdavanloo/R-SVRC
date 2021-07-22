function [ourM,problem]=sphere_methodsOther(X,M0)

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
        fprintf(' done (size: %d x %d) .\n', size(X));
    end
    [d,N]=size(X);
    assert(isreal(X),'X must be real.');
    
    if ~exist('inv_sig_0','var')|| isempty(M0) 
        M0=spherefactory(d,1).rand(); % initial_solution
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
    default.tolgradnorm=1e-6;
    default.stochastic=1;
%     
%     
%     %% Different noise_level: report last iterate
%     sigmafig=figure('Name','Different_noise_level');
%     options=default;
%     options.tolgradnorm=1e-8;
%     noise_level=[0.02,0.1,1,3];
%     
%     for line=1: 4
%        constant=noise_level(line);
%        noise=constant*randn(1,N);
%        flag=beta'*X+noise;
%        Y=2*(flag>0)-1;  
%        SNR=snr(beta'*X,noise)
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
    noise_level=[0.02,0.1,1,3];
    
    for line=1: 4
       constant=noise_level(line);
       noise=constant*randn(1,N);
       flag=beta'*X+noise;
       Y=2*(flag>0)-1;  
       SNR=snr(beta'*X,noise)
       
       rep_num=5;
       result_rep=zeros(rep_num,100);
       result_rep1=zeros(rep_num,100);
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