% Copyright © 2025 Yasaman Torabi, Shahram Shirani, James P. Reilly
%
% Cite this work as:
% Torabi, Yasaman; Shirani, Shahram; Reilly, James P. (2025),
% Large Language Model-based Nonnegative Matrix Factorization For Cardiorespiratory Sound Separation,
% arXiv preprint, https://doi.org/10.48550/arXiv.2502.05757.
%
% Reference:
% A. Cichocki, R. Zdunek, S. Choi, R. Plemmons, and S.-I. Amari.
% Novel multi-layer nonnegative tensor factorization with sparsity constraints.
% Springer LNCS, 4432:271-280, April 11-14, 2007.
%
% Original implementation written by Anh Huy Phan and Andrzej Cichocki (08-2008).


%%load data
[y1,fs1]=audioread ('heart.m4a');
[y2,fs2]=audioread ('lung.m4a');

h=y1(1:min(length(y1),length(y2)),1);
l=y2(1:min(length(y1),length(y2)),1);

%%Generate mixtures
X= zeros(2,length(l));
%%Time specifications:
   Fs = 800;                   % samples per second
   dt = 1/Fs;                   % seconds per sample
   StopTime = 0.25;             % seconds
   tt = (0:dt:StopTime-dt);     % seconds
%%Signals:

   X(1,:)=l';
   X(2,:)=h';

rng(100);
s = rng;
 [J,T] = size (X);
 I = 2; % number of mixtures
 A = rand (I,J);
 %%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% first round
 fprintf('first stage')
 Y = 0.2*A*X+1;
 
 %%Fatorize the source using ALPHA NMF over layers
 options = struct ('J',J,'alpha',2,'niter',1000,'verbose',0,...
 'algorithm','nmf_alpha','Nlayer',1);
 [AH,XH,PSNR]= nmf_multi_layer(Y,options ,X,1);

ac=xcorr(XH(1,:),XH(1,:));
[~,locs]=findpeaks(ac);
T1=mean(diff(locs)*0.1);

ac=xcorr(XH(2,:),XH(2,:));
[~,locs]=findpeaks(ac);
T2=mean(diff(locs)*0.1);

if (T1<T2)
 estimated_lung=XH(1,:);
 lung_PSNR=max(PSNR{1});
else
 estimated_lung=XH(2,:);
 lung_PSNR=max(PSNR{1});
end
%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% second round
 fprintf('second stage')

Y = 5*A*X+6;
 
 %%Fatorize the source using ALPHA NMF over layers
 options = struct ('J',J,'alpha',2,'niter',1000,'verbose',0,...
 'algorithm','nmf_alpha','Nlayer',1);
 [AH,XH ,PSNR]= nmf_multi_layer(Y,options ,X,1);

ac=xcorr(XH(1,:),XH(1,:));
[~,locs]=findpeaks(ac);
T1=mean(diff(locs)*0.1);

ac=xcorr(XH(2,:),XH(2,:));
[~,locs]=findpeaks(ac);
T2=mean(diff(locs)*0.1);

if (T1>T2)
 estimated_heart=XH(1,:);
 heart_PSNR=max(PSNR{1});
else
 estimated_heart=XH(2,:);
 heart_PSNR=max(PSNR{1});
end


%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Final Result

figure
subplot(2,2,1)
plot(X(1,:))
xlabel 'Time(s)'
   grid on
 title 'lung'
subplot(2,2,2)
plot(X(2,:))
   grid on
 title 'heart'
subplot(2,2,3)
plot(estimated_heart)
str = sprintf('estimated heart with PSNR= %0.2f', heart_PSNR)
title(str)
subplot(2,2,4)
plot(estimated_lung)
str = sprintf('estimated lung with PSNR= %0.2f', lung_PSNR)
title(str)

%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% functions
function [AH,XH,PSNR]= nmf_multi_layer (Y,options ,X,type )

if exist ('X','var') ~= 1, X = []; end
if exist ('type ','var') ~= 1, type = 1; end
PSNR = {}; AH = 1;
Yn = bsxfun ( @rdivide ,Y,max (Y,[],2));
for l = 1: options.Nlayer
if l>1
Yl = XH;
else
Yl = Y;
end
[Al,XH] = nmf_alpha ( Yl,options );
if any ( isnan (Al (:))), break , end
AH = AH*Al;
if ~isempty (X)
if type ==1 %XPN
Xref = XH;
else
Xref = AH';
end
Xref = bsxfun ( @rdivide ,Xref ,max(Xref ,[],2));
PSNR {l} = PSNRNMF (X,Xref);
Yhat = AH*XH;
Yhat = bsxfun ( @rdivide ,Yhat ,max(Yhat ,[],2));
end
end % for l
end


function [A,X] = nmf_alpha (Y,options )
defopts = struct ('J',size (Y,1),'algtype',[1 1],'init',[1 1],...
'nrestart',1,'niter',1000,'tol',1e-5,'alpha',2,'omega',[1 1],...
'lsparse',[0 0],'lortho',[0 0],'alphaS',0,'A0',[],'X0',[],'verbose',0);
if ~exist ('options','var'), options = struct; end
[J,algtype ,init ,nrestart ,niter ,tol ,alpha ,w,lsparse ,...
lcorr ,alphaS ,A0,X0,verbose ] = scanparam ( defopts ,options );
algname = { 'Alpha ' 'DualKL ','Hellinger ','KL ','Pearson ','ALS ','Fix '
alpha , 0, .5 1 2 0 0};
if ( alpha == 0), algtype ( algtype == 1) = 2; end % change to D−KL
%alpha =  algname {2, algtype(1,1) };
alpha=[2 2];
Y(Y< 0) = eps;
nr_best = 0;
No_iter = 30; % number of iterations to select the best trial
%%Main process
for nr= 0: nrestart
if (nr == nrestart )&&( nrestart > 0) % initialize
A = A_best; X = X_best; No_iter = niter;
else
[A,X] = nmf_initialize (Y,J ,{A0 X0'}); X = X';
end
cost = costfunction;
for k = 1: No_iter
X = nmfcore (Y,A,X,algtype (2), alpha (2), lsparse (2), lcorr (2),w(2));
A = nmfcore (Y',X',A',algtype (1), alpha (1), lsparse (1), lcorr (1),w(1))';
if ( algtype (2) ~= 7)
A = normalize (A);

elseif ismember ( algtype (1),[1 3 4 5]) % fix scaling
A = bsxfun( @rdivide ,A,sum (X,2).^(w (1)*(1+ alphaS )/ alpha (1)));
end
if (nr == nrestart ) && (( mod (k,30)==0) || (k == No_iter ))
stop=checkstoppingcondition; % stopping condition
if verbose
fprintf (1,'Best trial %d, step %d, Cost value %d\n',...
nr_best +1,k,cost);
end
if stop , break; end
end
end % k
if (nr < nrestart ) % select best trial
cost = costfunction;
if (nr == 0) || ( cost < cost_min )
A_best = A; X_best = X; cost_min = cost; nr_best = nr;
end
if verbose
fprintf (1, 'Trial %d, Cost value = %e\n', nr+1, cost);
end
end
end % nr
function X = nmfcore (Y,A,X,type_alg ,alpha , lsparse ,lcorr ,w)
switch type_alg
case {1, 3 ,4,5} % Alpha−divergence , Hellinger , KL, Pearson
Jor = 0;
if lcorr
Jor = lcorr *bsxfun (@minus ,X,sum (X,1));
end
X = (X.*(A'*(Y./(A*X + eps )).^ alpha - lsparse + Jor) ...
.^(w/ alpha )).^(1+ alphaS );
case 2 % alpha = 0, D−KL
X = (X.*exp (w* bsxfun ( @rdivide ,A,sum (A,1))*...
log (Y./(A*X+ eps )+ eps ))).^(1+ alphaS );
case 6 % ALS
X = pinv (A)*Y;
end
X = max (1E6*eps,X);
end
function stop=checkstoppingcondition
cost_old = cost; cost = costfunction;
stop = abs( cost_old-cost ) <= tol*cost;
end
function cost = costfunction
Yhat = A*X+ eps;
if ( alpha (1) == alpha (2)) && ( alpha (1) ~= 0) && ( alpha (1) ~= 1)
cost = sum(sum(Y.*(( Y./ Yhat ).^( alpha(1)-1)-1)...
/( alpha (1)*( alpha (1)-1)) - (Y-Yhat )/ alpha (1)));
else
cost = sum(sum(Y.*log(Y./ Yhat + eps) - Y + Yhat ));
end
end
function A = normalize (A)
A = bsxfun ( @rdivide ,A,sum (A));
end
end

function varargout = scanparam ( defoptions ,options )
allfields = fieldnames ( defoptions );
opts = defoptions;
for k = 1: numel ( allfields )
if isfield ( options ,allfields {k})
if numel ( options. ( allfields {k}))<numel ( defoptions. ( allfields {k }))
opts. ( allfields {k }) = repmat ( options. ( allfields {k}),...
1,numel ( defoptions. ( allfields {k })));
else
opts. ( allfields {k }) = options. ( allfields {k});
end
end
end
if nargout > 1
varargout = struct2cell ( opts);
else
varargout = opts;
end
end

function varargout = nmf_initialize(Y, J, Fact)
factorsize = size(Y);
for k = 1:2
        Fact{k} = rand(factorsize(k), J);
end
varargout = Fact;

end


function PSNR = PSNRNMF (X,XH)
X1=X(1,:)-mean(X(1,:));
X1=X1./max(abs(X1));
XH1=XH(1,:)-mean(XH(1,:));
XH1=XH1./max(abs(XH1));
[J,T] = size (X1);
PSNR11 = zeros (J,1);
for j = 1:J
PSNR11 (j)=  sum ( bsxfun ( @minus ,XH1,X1(j,:)).^2,2);
end
PSNR11 = -10*log10 (( PSNR11 + eps )/T);

X1=X(1,:)-mean(X(1,:));
X1=X1./max(abs(X1));
XH2=XH(2,:)-mean(XH(2,:));
XH2=XH2./max(abs(XH2));
[J,T] = size (X1);
PSNR12 = zeros (J,1);
for j = 1:J
PSNR12 (j)=  sum ( bsxfun ( @minus ,XH2,X1(j,:)).^2,2);
end
PSNR12 = -10*log10 (( PSNR12 + eps )/T);

X2=X(2,:)-mean(X(2,:));
X2=X2./max(abs(X2));
XH1=XH(1,:)-mean(XH(1,:));
XH1=XH1./max(abs(XH1));
[J,T] = size (X2);
PSNR21 = zeros (J,1);
for j = 1:J
PSNR21 (j)=  sum ( bsxfun ( @minus ,XH1,X2(j,:)).^2,2);
end
PSNR21 = -10*log10 (( PSNR21 + eps )/T);

X2=X(2,:)-mean(X(2,:));
X2=X2./max(abs(X2));
XH2=XH(2,:)-mean(XH(2,:));
XH2=XH2./max(abs(XH2));
[J,T] = size (X2);
PSNR22 = zeros (J,1);
for j = 1:J
PSNR22 (j)=  sum ( bsxfun ( @minus ,XH2,X2(j,:)).^2,2);
end
PSNR22 = -10*log10 (( PSNR22 + eps )/T);

PSNR=[PSNR11 PSNR12 PSNR21 PSNR22];

end
