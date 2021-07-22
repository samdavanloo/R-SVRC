function [x, cost, info, options]=rsvrc(problem,x,options)

    
    M=problem.M;
    N=problem.ncostterms;
    tensor_factory=problem.tensor_factory; 
    subproblem_cubic.M=problem.M;
    
   
    partialehess_tensor=problem.partialehess_tensor;
    partialegrad=problem.partialegrad;
    
    %Vefify that the problem description is sufficient for the solver.
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
                'No cost provided. The algorithm will likely abort.');
    end
    if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
        % Note: we do not give a warning if an approximate gradient is
        % explicitly given in the problem description, as in that case the
        % user seems to be aware of the issue.
        warning('manopt:getGradient:approx', ['No gradient provided. ' ...
                'Using an FD approximation instead (slow).\n' ...
                'It may be necessary to increase options.tolgradnorm.\n'...
                'To disable this warning: ' ...
                'warning(''off'', ''manopt:getGradient:approx'')']);
        problem.approxgrad = approxgradientFD(problem);
    end
    if ~canGetHessian(problem) && ~canGetApproxHessian(problem)
        % Note: we do not give a warning if an approximate Hessian is
        % explicitly given in the problem description, as in that case the
        % user seems to be aware of the issue.
        warning('manopt:getHessian:approx', ['No Hessian provided. ' ...
                'Using an FD approximation instead.\n' ...
                'To disable this warning: ' ...
                'warning(''off'', ''manopt:getHessian:approx'')']);
        problem.approxhess = approxhessianFD(problem);
    end
    
    % Set local defaults 
    localdefaults.maxepoch=20; 
    localdefaults.maxinneriter=5;
    localdefaults.tolgradnorm=1e-6;
    localdefaults.random_seed=1;
    
    localdefaults.batchsize_g=N/20; 
    localdefaults.batchsize_h=N/20;
    localdefaults.stochastic=1;

    localdefaults.storedepth = 2;
    localdefaults.subproblemsolver = @arc_conjugate_gradient;
    localdefaults.tochastic=1;
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    rng(options.random_seed);
    
    if ~isfield(options, 'sigma_0')
        if isfield(M, 'typicaldist')
            options.sigma_0 = 200/M.typicaldist();
        else
            options.sigma_0 = 200/sqrt(M.dim());
        end 
    end

    batchsize_g=options.batchsize_g;
    batchsize_h=options.batchsize_h;

    % if no initial point x is given by the user, generate one at random.
    if ~exist('x','var') || isempty(x)
        x=problem.M.rand();
    end

    % Create a store database and get a key for the current x. 
    storedb=StoreDB(options.storedepth);
    key=storedb.getNewKey();
    
    % Compute objective-related quantities for x. 
    %cost=getCost(problem,x,storedb,key);
    %grad=getGradient(problem,x,storedb,key);
    cost=problem.cost(x);
    grad=M.egrad2rgrad(x, partialegrad(x,1:N));
    gradnorm=M.norm(x,grad);
    
    lambda=NaN;
    %[u, lambda] = hessianextreme(problem, x, 'min');
    
    % Initialize regularization parameter. 
    sigma=options.sigma_0;

    iter_count=0;
    so_count=0;
    
    % Save stats in a struct array info, and preallocate. 
    stats=savestats(problem,x,storedb,key,options);
    info(1)=stats;
    info(min(10000,(options.maxepoch+1)*(options.maxinneriter+1))).iter=[];
    
    x_epoch=x;
    perm_idx_g=randi(N,1,batchsize_g*options.maxinneriter*options.maxepoch);
    perm_idx_h=randi(N,1,batchsize_h*options.maxinneriter*options.maxepoch);
    
    stochastic=options.stochastic;
    elap_time=0;
    
    if stochastic==0
        %% For the deterministic setting 
        for epoch =1: options.maxepoch
            x_itr=x_epoch;
            for itr=1:options.maxinneriter    
                t1=cputime;
                H2=partialehess_tensor(x_itr,1:N);   
                g2=partialegrad(x_itr,1:N);
                                       
                v=M.egrad2rgrad(x_itr, g2);
                v_norm=M.norm(x_itr,v);
                
                subproblem_cubic.hess=@(x,u)M.ehess2rhess(x, g2, tensor_factory(H2,u), u);

                        [eta_cubic, Heta, hesscalls, stop_str, substats] = ...
                options.subproblemsolver(subproblem_cubic, x_itr, v, v_norm, ...
                                             sigma, options, storedb, key);
                                                      
                so_count=so_count+1;
                newx = M.exp(x_itr,eta_cubic);
                
                elap_time=elap_time+cputime-t1;
                          
                newkey = storedb.getNewKey();
                
                newcost=problem.cost(newx);
 
                newgrad=getGradient(problem,newx);
                
                %[u, lambda] = hessianextreme(problem, newx, 'min');

                x_itr = newx;
                key = newkey;
                cost = newcost;
                grad = newgrad;
                gradnorm = M.norm(x_itr, grad);
              
                % iter is the number of iterations we have completed.
                iter_count = iter_count + 1;

                % Make sure we don't use too much memory for the store database.
                storedb.purge();

                % Log statistics for freshly executed iteration.
                stats = savestats(problem, x_itr, storedb, key, options);
                info(iter_count+1) = stats;

                [stop, reason] = stoppingcriterion(problem, x_itr, options, ...
                                                             info, iter_count+1);
                if stop
                    if options.verbosity >= 1
                            fprintf(['\n' reason '\n']);
                    end
                    break;
                end
            end
            x_epoch=x_itr;
            if stop
                break;
            end
        end  
    else
        %% for the stochastic setting 
        for epoch =1: options.maxepoch
            
            t1=cputime;
            eg_epoch=partialegrad(x_epoch,1:N);        
            rg_epoch=M.egrad2rgrad(x_epoch,eg_epoch);    
            so_count=so_count+1;   
            H_epoch=partialehess_tensor(x_epoch,1:N);
            elap_time=elap_time+cputime-t1;
  
            x_itr=x_epoch;
            for itr=1:options.maxinneriter
                t2=cputime;
                if itr==1
                    v=rg_epoch;
                    v_norm=M.norm(x_itr,v);
                    subproblem_cubic.hess=@(x,u)M.ehess2rhess(x, eg_epoch, tensor_factory(H_epoch,u), u);
                    
                        [eta_cubic, Heta, hesscalls, stop_str, substats] = ...
                options.subproblemsolver(subproblem_cubic, x_itr, v, v_norm, ...
                                             sigma, options, storedb, key);
                
                else
                    
                    start_index_g= (iter_count) * batchsize_g +1;
                    end_index_g=(iter_count+1)*batchsize_g;
                    idx_bg=perm_idx_g(start_index_g:end_index_g);
                    start_index_h= (iter_count) * batchsize_h +1;
                    end_index_h=(iter_count+1)*batchsize_h;
                    idx_bh=perm_idx_h(start_index_h:end_index_h);

                    eta=M.log(x_epoch,x_itr);
                    v1=M.paralleltransp(x_epoch,x_itr,rg_epoch);
                    
                    v2=(N/batchsize_g)*getPartialGradient(problem, x_itr, idx_bg);

                    ev3_0=(N/batchsize_g)*getPartialEuclideanGradient(problem, x_epoch, idx_bg);
                    v3_0=M.egrad2rgrad(x_epoch,ev3_0);
                    v3=M.paralleltransp(x_epoch,x_itr,v3_0);

                    v4_0=tensor_factory((N/batchsize_g)*partialehess_tensor(x_epoch,idx_bg),eta);
                    v4_1=M.ehess2rhess(x_epoch, ev3_0, v4_0, eta);
                    v4_2=M.paralleltransp(x_epoch,x_itr,v4_1);

                    v5_0=tensor_factory(H_epoch,eta);
                    v5_1=M.ehess2rhess(x_epoch,eg_epoch,v5_0,eta);
                    v5_2=M.paralleltransp(x_epoch,x_itr,v5_1);

                    v=v1+v2-v3-v4_2+v5_2;
                    v_norm=M.norm(x_itr,v);

                    H2=(N/batchsize_h)*partialehess_tensor(x_itr,idx_bh);
                    H3=(N/batchsize_h)*partialehess_tensor(x_epoch,idx_bh);
                    g2=(N/batchsize_h)*getPartialEuclideanGradient(problem, x_itr, idx_bh);
                    g3=(N/batchsize_h)*getPartialEuclideanGradient(problem, x_epoch, idx_bh);


                    subproblem_cubic.hess=@subproblem_hess;
                            [eta_cubic, Heta, hesscalls, stop_str, substats] = ...
                    options.subproblemsolver(subproblem_cubic, x_itr, v, v_norm, ...
                                                 sigma, options, storedb, key);

                    so_count=so_count+min(N,batchsize_h+batchsize_g)/N;
                    
                end
                newx = M.exp(x_itr,eta_cubic);
                elap_time=elap_time+cputime-t2;
                
                tic;
                newkey = storedb.getNewKey();
                newcost=getCost(problem,newx);
                newgrad=getGradient(problem,newx);
                toc
                

                %[u, lambda] = hessianextreme(problem, newx, 'min');
                x_itr = newx;
                key = newkey;
                cost = newcost;
                grad = newgrad;
                gradnorm = M.norm(x_itr, grad);

                % iter is the number of iterations we have completed.
                iter_count = iter_count + 1;

                % Make sure we don't use too much memory for the store database.
                storedb.purge();

                % Log statistics for freshly executed iteration.
                stats = savestats(problem, x_itr, storedb, key, options);
                info(iter_count+1) = stats;
                [stop, reason] = stoppingcriterion(problem, x_itr, options, ...
                                                             info, iter_count+1);
                if stop
                    if options.verbosity >= 1
                            fprintf(['\n' reason '\n']);
                    end
                    break;
                end
            end
            x_epoch=x_itr;
            if stop
                break;
            end
        end
    end
        
    info = info(1:iter_count+1);
    x=x_itr;
    
    function h = subproblem_hess(x, u)   
        u_transported=M.paralleltransp(x,x_epoch,u);
        h1_0=tensor_factory(H_epoch,u_transported);
        h1_1=M.ehess2rhess(x_epoch, eg_epoch, h1_0, u_transported);
        h1_2=M.paralleltransp(x_epoch,x,h1_1);
        
        h2_0=tensor_factory(H2,u);
        h2_1=M.ehess2rhess(x, g2, h2_0, u);
        
        h3_0=tensor_factory(H3,u_transported);
        h3_1=M.ehess2rhess(x_epoch, g3, h3_0, u_transported);
        h3_2=M.paralleltransp(x_epoch,x,h3_1);
        
        h=h1_2+h2_1-h3_2;        
    end


    % Routine in charge of collecting the current iteration statistics
    function stats =savestats(problem, x, storedb,key,options)
        
            stats.iter=iter_count;
            stats.cost=cost;
            stats.gradnorm=gradnorm;
            stats.sigma=sigma;
            stats.lambda_min=lambda;
            stats.so_count=so_count;
            
            if iter_count==0
                stats.hessiancalls=0;
                stats.time_cpu=0; 
                stats.x_itr=x;
                %stats.subproblem=struct(); %In case we want to store the detailed information from subproblem solver   
            else
                stats.hessiancalls = hesscalls;
                stats.time_cpu=elap_time;
                stats.x_itr=x_itr;
                %stats.subproblem=struct(); 
            end
            stats=applyStatsfun(problem,x,storedb,key,options,stats);
    end
end