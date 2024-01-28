function [a,e,BIC,c] = ARX_LS_Estimation(x,y,p,q,option)
% Model parameter estiamtion method

% Input parameters:
% x, observations, column vector, with length of N
% y, external Xgenerous input, column vector, speed or load with length of N
% p, na should be a scalar
% q, nx should be a scalar, 

% Output parameters:  
% a should be a vector
% c should be a vector
% BIC scalar

switch option
case 'AR'
    if iscell(x)
        PHI = [];
        xl = [];
        for ii = 1:length(x)
            xi = x{ii}';
            PHI = [PHI; PredsConst(xi,p)];
            xl = [xl; xi];
        end
        a = (PHI'*PHI)\(PHI'*xl);
        N = length(xl);
        e = xl - PHI*a;
    else
        PHI = PredsConst(x,p);
        a = (PHI'*PHI)\(PHI'*x);
        N = length(x);
        e = x - PHI*a;
    end
    %a = lscov(PHI,x);
    
    %
    %options = optimoptions('fmincon');
    %options.StepTolerance  = 1e-50;
    %fun = @(a)sum((x - PHI*a).^2);
    %a = fmincon(fun,a0,[],[],[],[],[],[],[],options);    
    BIC = N*log(mse(e)) + log(N)*p;
    %AIC = log(mse(e)) + 2*p/N;
    c = [];

case 'ARX'
    N = length(x);
    PHI = zeros(N,p+q+1);
    x_lay = zeros(p,1);
    y_lay = zeros(q+1,1);
    theta = zeros(p+q+1,1); %#ok<PREALL>

    for i = 1:N
        if i > p
            x_lay = x(i-1:-1:i-p,1);
        elseif i <= p
            for j = 1:p
                if (i-j)<1
                    x_lay(j,1) = 0;
                else
                    x_lay(j,1) = x(i-j,1);
                end
            end 
        end
        if i > q
            y_lay = y(i:-1:i-q,1);
        elseif i <= q
            for j = 0:q
                if (i-j)<1
                    y_lay(j+1,1) = 0;
                else
                    y_lay(j+1,1) = y(i-j,1);
                end
            end 
        end  
        PHI(i,:) = -[x_lay' y_lay'];
    end

    %theta = lscov(PHI,x);
    theta = (PHI'*PHI)\(PHI'*x);
    e = x - PHI*theta;

    a = theta(1:p);
    c = theta(p+1:end);

    BIC = N*log(mse(e)) + log(N)*(p+q+1);
    
    case 'SSAR'
    m = p;
    N = length(x);
    PHI = zeros(N,2*m-1);
    x_lay = zeros(2*m-1,1);
    a = zeros(m,1); 
    for i = 1:N
        if i > (2*m-1)
            x_lay = x(i-1:-1:(i-(2*m-1)),1);
        elseif i <= (2*m-1)
            for j = 1:(2*m-1)
                if (i-j)<1
                    x_lay(j,1) = 0;
                else
                    x_lay(j,1) = x(i-j,1);
                end
            end 
        end
        PHI(i,:) = -[x_lay'];
    end
    
    V = zeros(m);
    for j = 1:m-1
        V(m-j,j) = 1;       
    end
    V = [diag(ones(1,m)) V(:,1:(m-1))];
    
    %x = x - circshift(x,2*m);
    
    PHI = PHI*V';
    %a = lscov(PHI,x);
    a = (PHI'*PHI)\(PHI'*x);
    e = x - PHI*a;
    BIC = N*(log(2*pi)+1) + N*log(sum(e.^2)/N) + log(N)*(p);
    c = [];   
end
end

function PHI = PredsConst(x,p)
% Creates a PHI for a single segement of x
N = length(x);
PHI = zeros(N,p);
x_lay = zeros(p,1);
a0 = zeros(p,1); 
for i = 1:N
    if i > p
        x_lay = x(i-1:-1:i-p,1);
    elseif i <= p
        for j = 1:p
            if (i-j)<1
                x_lay(j,1) = 0;
            else
                x_lay(j,1) = x(i-j,1);
            end
        end 
    end
    PHI(i,:) = -[x_lay'];
end
end