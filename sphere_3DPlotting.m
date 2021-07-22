function [ourM,problem]=sphere_3DPlotting(X,M0)

    %% Data simulation 
  
    rng(4);
    % If no inputs are provided, since this is an example file, simulate 
    % the data set: n samples in R^d. This is for illustration purposes only.
   
    if ~exist('X','var')||isempty(X)
        d=3;
        N=1e3;
        fprintf('Generating data...');  
        
        % randomly generate the true beta. 
        %beta=spherefactory(d,1).rand();
        beta=[1;0;0];
        Sigma=[2 1 0;0 5 1; -2 -3 10];
        Sigma=Sigma+Sigma';
        mu=zeros(d,1);
        X=mvnrnd(mu,Sigma,N)';
       
        constant=4;
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



    %% Iterates generating
    options.maxinneriter=5;
    options.sigma_0=5;
    options.batchsize_g=N/10;
    options.batchsize_h=N/10;
    options.tolgradnorm=1e-6;
    
    [ourM,ourcost, our_info, ouroptions]=rsvrc(problem,M0,options);
    
    x_save=[our_info.x_itr];
    f_temp = size(x_save, 2);
    for k = 1:size(x_save, 2)
        f_temp(k) = problem.cost(x_save(:, k));
    end

    %% Visiualization

    res = 201;
   
    lambda = linspace(-pi, pi, res);
    theta = linspace(-pi/2, pi/2, ceil(res/2));
    [L, T] = meshgrid(lambda, theta);

    [X_plot, Y_plot, Z_plot] = sph2cart(L, T, 1); % transfer to X,Y,Z coordinate
    [m, n] = size(X_plot);

    z = zeros(m, n, 3);
    z(:, :, 1) = X_plot;
    z(:, :, 2) = Y_plot;
    z(:, :, 3) = Z_plot;

    f_temp = zeros(m, n);
    for i_contur = 1:m * n
        [a_temp, b_temp] = ind2sub([m, n], i_contur);
        f_temp(i_contur) = problem.cost([z(a_temp, b_temp, 1); z(a_temp, b_temp, 2); z(a_temp, b_temp, 3)]);
    end


    lvls = 20; % number of contour lines
    figure(1)
    clf
    subplot(1, 2, 1)
    mesh(L, T, f_temp)


    c_f = contour(L, T, f_temp, lvls, 'ShowText', 'on');
    [~, cl] = size(c_f);
    clf;

    surf(X_plot, Y_plot, Z_plot, 'FaceColor', [0.93, 0.93, 0.93], 'FaceAlpha', 0.5, 'EdgeColor', 'none'); % Create a solid sphere to plot the results on
    hold on;
    cmap = jet(lvls); % Colormap for the contours
    k = 1;
    cnum = 1;
    clvl = c_f(1, k);
    cmin = clvl;
    while k < cl % Draw each contour line.
        kl = c_f(2, k);
        v = k + 1:k + kl;
        xv = cos(c_f(2, v)) .* cos(c_f(1, v));
        yv = cos(c_f(2, v)) .* sin(c_f(1, v));
        zv = sin(c_f(2, v));
        if c_f(1, k) ~= clvl
            cnum = cnum + 1;
            clvl = c_f(1, k);
        end
        plot3(xv, yv, zv, '-', 'linewidth', 1, 'Color', cmap(cnum, :)), hold on;
        k = k + kl + 1;
    end
    %title('3D Contour plot of f on the sphere')
    colormap(cmap);
    colorbar;
    caxis([cmin, clvl]);
    axis tight; daspect([1 1 1]);

    xlabel('x')
    ylabel('y')

    %% Plot iterations

    hold on
    plot3(x_save(1, :), x_save(2, :), x_save(3, :), 'b-', 'LineWidth', 2, 'MarkerSize', 10)
    dir = x_save(:, 2:end) - x_save(:, 1:end-1);
    quiver3(x_save(1, 1:end-1), x_save(2, 1:end-1), x_save(3, 1:end-1), dir(1, :), dir(2, :), dir(3, :), 'r', 'MaxHeadSize', 30, 'LineWidth', 2);

    
end