%midterm exam
%Kei Hirano

clear all;

%*******************************************************************
%**      assign names to parameters
%********************************************************************/
alpha = 0.5;
sigma = 1.5;
omega = 0.5;
beta = 0.99;
kappa = 0.5;
psi_pi = 1.5;
psi_y = 0.5;
rho_a = 0.5;
rho_p = 0.5;

%********************************************************************
%**      matrices of canonical system
%********************************************************************

%/** equation indices **/
eq_is     = 1;
eq_pc     = 2;
eq_rule   = 3;
eq_za     = 4;
eq_zp     = 5;
eq_pi1    = 6;
eq_pi2    = 7;
eq_Ey     = 8;
eq_Epi    = 9;

%/** variable indices **/
v_y     = 1;
v_pi    = 2;
v_r     = 3;
v_za    = 4;
v_zp    = 5;
v_pi1   = 6;
v_pi2   = 7;
v_Ey    = 8;
v_Epi   = 9;

%/** shock indices **/
e_r = 1;
e_a = 2;
e_p = 3;

%/** expectation error indices **/
n_y  = 1;
n_pi = 2;

%/** summary **/
neq  = 9;
neps = 3;
neta = 2;

%/** initialize matrices **/
GAM0 = zeros(neq,neq);
GAM1 = zeros(neq,neq);
   C = zeros(neq,1);
 PSI = zeros(neq,neps);
 PPI = zeros(neq,neta);

 %/**********************************************************
%**      1. IS curve 
%**********************************************************/
GAM0(eq_is,v_y)   =  1;
GAM0(eq_is,v_r)  = alpha/sigma;
GAM0(eq_is,v_za)   =  -1;
GAM0(eq_is,v_Ey) = -alpha;
GAM0(eq_is,v_Epi)  = -alpha/sigma;
GAM1(eq_is,v_y)   =  1-alpha;

%/**********************************************************
%**      2. Phillips curve 
%**********************************************************/
GAM0(eq_pc,v_y)  = -kappa;
GAM0(eq_pc,v_pi) = 1+beta*omega;
GAM0(eq_pc,v_zp)   = -1;
GAM0(eq_pc,v_Epi)  = -beta;
GAM1(eq_pc,v_pi)  = omega;

%/**********************************************************
%**      3. Monetary policy rule 
%**********************************************************/
GAM0(eq_rule,v_y)  = -psi_y;
GAM0(eq_rule,v_pi) = -psi_pi/4;
GAM0(eq_rule,v_r)  = 1;
GAM1(eq_rule,v_r)  = psi_pi/4;
GAM1(eq_rule,v_pi1)  = psi_pi/4;
GAM1(eq_rule,v_pi2)  = psi_pi/4;
PSI(eq_rule,e_r)   = 1;

%/**********************************************************
%**      4.- 5. Shock Processes 
%**********************************************************/
GAM0(eq_za,v_za)  = 1;
GAM1(eq_za,v_za)  = rho_a;
PSI(eq_za,e_a)    = 1;

GAM0(eq_zp,v_zp)  = 1;
GAM1(eq_zp,v_zp)  = rho_p;
PSI(eq_zp,e_p)    = 1;

%/**********************************************************
%**      6.- 7. Identity equations
%**********************************************************/
GAM0(eq_pi1,v_pi1) = 1;
GAM1(eq_pi1,v_pi)  = 1;

GAM0(eq_pi2,v_pi2) = 1;
GAM1(eq_pi2,v_pi1) = 1;

%/**********************************************************
%**      8.- 9. Expectation errors
%**********************************************************/
GAM0(eq_Ey,v_y)  = 1;
GAM1(eq_Ey,v_Ey) = 1;
 PPI(eq_Ey,n_y)  = 1;

GAM0(eq_Epi,v_pi)  = 1;
GAM1(eq_Epi,v_Epi) = 1;
 PPI(eq_Epi,n_pi)  = 1;

%/********************************************************************
%**  Solve the model: QZ(generalized Schur) decomposition by GENSYS
%********************************************************************/
[PHI1,C,PHIe,fmat,fwt,ywt,gev,eu] = gensys(GAM0,GAM1,C,PSI,PPI) 

%********************************************************************
%**  Compute covariance by dlyap.m
%********************************************************************

%Covariance matrix for shocks.
sig_a = 0.5;
sig_p = 0.5;
sig_r = 0.1;
Sigeps = zeros(neps,neps);
Sigeps(e_a,e_a) = sig_a^2;
Sigeps(e_p,e_p) = sig_p^2;
Sigeps(e_r,e_r) = sig_r^2;

Sigs = dlyap(PHI1,PHIe,Sigeps)

%Calculate correlation coefficient between y and pi
cov_ypi = Sigs(1, 2);
var_y = Sigs(1,1);
var_pi = Sigs(2,2);
r_ypi = cov_ypi/sqrt(var_y*var_pi);
disp(["r_ypi:",num2str(r_ypi)]);


%Find optimal Psi
objective = @(psi) loss(psi, alpha, sigma, omega, beta, kappa, rho_a, rho_p, sig_a, sig_p, sig_r);
psi_ini = [0.5; 1.5];
options = optimset('Display', 'iter')
[opt_psi, fval] = fminsearch(objective, psi_ini, options);

disp('Optimal psi_y and psi_pi:');
disp(opt_psi);

% ======================================
% Function: loss
% ======================================
function sumvar = loss(psi, alpha, sigma, omega, beta, kappa, rho_a, rho_p, sig_a, sig_p, sig_r) 
psi_y = psi(1);
psi_pi = psi(2);

%********************************************************************
%**      matrices of canonical system
%********************************************************************


%/** equation indices **/
eq_is     = 1;
eq_pc     = 2;
eq_rule   = 3;
eq_za     = 4;
eq_zp     = 5;
eq_pi1    = 6;
eq_pi2    = 7;
eq_Ey     = 8;
eq_Epi    = 9;

%/** variable indices **/
v_y     = 1;
v_pi    = 2;
v_r     = 3;
v_za    = 4;
v_zp    = 5;
v_pi1   = 6;
v_pi2   = 7;
v_Ey    = 8;
v_Epi   = 9;

%/** shock indices **/
e_r = 1;
e_a = 2;
e_p = 3;

%/** expectation error indices **/
n_y  = 1;
n_pi = 2;

%/** summary **/
neq  = 9;
neps = 3;
neta = 2;

%/** initialize matrices **/
GAM0 = zeros(neq,neq);
GAM1 = zeros(neq,neq);
   C = zeros(neq,1);
 PSI = zeros(neq,neps);
 PPI = zeros(neq,neta);

 %/**********************************************************
%**      1. IS curve 
%**********************************************************/
GAM0(eq_is,v_y)   =  1;
GAM0(eq_is,v_r)  = alpha/sigma;
GAM0(eq_is,v_za)   =  -1;
GAM0(eq_is,v_Ey) = -alpha;
GAM0(eq_is,v_Epi)  = -alpha/sigma;
GAM1(eq_is,v_y)   =  1-alpha;

%/**********************************************************
%**      2. Phillips curve 
%**********************************************************/
GAM0(eq_pc,v_y)  = -kappa;
GAM0(eq_pc,v_pi) = 1+beta*omega;
GAM0(eq_pc,v_zp)   = -1;
GAM0(eq_pc,v_Epi)  = -beta;
GAM1(eq_pc,v_pi)  = omega;

%/**********************************************************
%**      3. Monetary policy rule 
%**********************************************************/
GAM0(eq_rule,v_y)  = -psi_y;
GAM0(eq_rule,v_pi) = -psi_pi/4;
GAM0(eq_rule,v_r)  = 1;
GAM1(eq_rule,v_r)  = psi_pi/4;
GAM1(eq_rule,v_pi1)  = psi_pi/4;
GAM1(eq_rule,v_pi2)  = psi_pi/4;
PSI(eq_rule,e_r)   = 1;

%/**********************************************************
%**      4.- 5. Shock Processes 
%**********************************************************/
GAM0(eq_za,v_za)  = 1;
GAM1(eq_za,v_za)  = rho_a;
PSI(eq_za,e_a)    = 1;

GAM0(eq_zp,v_zp)  = 1;
GAM1(eq_zp,v_zp)  = rho_p;
PSI(eq_zp,e_p)    = 1;

%/**********************************************************
%**      6.- 7. Identity equations
%**********************************************************/
GAM0(eq_pi1,v_pi1) = 1;
GAM1(eq_pi1,v_pi)  = 1;

GAM0(eq_pi2,v_pi2) = 1;
GAM1(eq_pi2,v_pi1) = 1;

%/**********************************************************
%**      8.- 9. Expectation errors
%**********************************************************/
GAM0(eq_Ey,v_y)  = 1;
GAM1(eq_Ey,v_Ey) = 1;
 PPI(eq_Ey,n_y)  = 1;

GAM0(eq_Epi,v_pi)  = 1;
GAM1(eq_Epi,v_Epi) = 1;
 PPI(eq_Epi,n_pi)  = 1;

%/********************************************************************
%**  Solve the model: QZ(generalized Schur) decomposition by GENSYS
%********************************************************************/
[PHI1,~,PHIe] = gensys(GAM0,GAM1,C,PSI,PPI) 
Sigeps = zeros(neps,neps);
Sigeps(e_a,e_a) = sig_a^2;
Sigeps(e_p,e_p) = sig_p^2;
Sigeps(e_r,e_r) = sig_r^2;


Sigs = dlyap(PHI1, PHIe, Sigeps);

var_y = Sigs(1,1);
var_pi = Sigs(2,2);
var_r = Sigs(3,3);

sumvar = var_pi + var_y + var_r;
end

% ======================================
% Function: gensys
% ======================================

function [G1,C,impact,fmat,fwt,ywt,gev,eu,loose]=gensys(g0,g1,c,psi,pi,div)
% function [G1,C,impact,fmat,fwt,ywt,gev,eu,loose]=gensys(g0,g1,c,psi,pi,div)
% System given as
%        g0*y(t)=g1*y(t-1)+c+psi*z(t)+pi*eta(t),
% with z an exogenous variable process and eta being endogenously determined
% one-step-ahead expectational errors.  Returned system is
%       y(t)=G1*y(t-1)+C+impact*z(t)+ywt*inv(I-fmat*inv(L))*fwt*z(t+1) .
% If z(t) is i.i.d., the last term drops out.
% If div is omitted from argument list, a div>1 is calculated.
% eu(1)=1 for existence, eu(2)=1 for uniqueness.  eu(1)=-1 for
% existence only with not-s.c. z; eu=[-2,-2] for coincident zeros.
% By Christopher A. Sims
% Corrected 10/28/96 by CAS
eu=[0;0];
realsmall=1e-6;
fixdiv=(nargin==6);
n=size(g0,1);
[a b q z v]=qz(g0,g1);
if ~fixdiv, div=1.01; end
nunstab=0;
zxz=0;
for i=1:n
% ------------------div calc------------
   if ~fixdiv
      if abs(a(i,i)) > 0
         divhat=abs(b(i,i))/abs(a(i,i));
	 % bug detected by Vasco Curdia and Daria Finocchiaro, 2/25/2004  A root of
	 % exactly 1.01 and no root between 1 and 1.02, led to div being stuck at 1.01
	 % and the 1.01 root being misclassified as stable.  Changing < to <= below fixes this.
         if 1+realsmall<divhat & divhat<=div
            div=.5*(1+divhat);
         end
      end
   end
% ----------------------------------------
   nunstab=nunstab+(abs(b(i,i))>div*abs(a(i,i)));
   if abs(a(i,i))<realsmall & abs(b(i,i))<realsmall
      zxz=1;
   end
end
div ;
nunstab;
if ~zxz
   [a b q z]=qzdiv(div,a,b,q,z);
end
gev=[diag(a) diag(b)];
if zxz
   disp('Coincident zeros.  Indeterminacy and/or nonexistence.')
   eu=[-2;-2];
   % correction added 7/29/2003.  Otherwise the failure to set output
   % arguments leads to an error message and no output (including eu).
   G1=[];C=[];impact=[];fmat=[];fwt=[];ywt=[];gev=[];
   return
end
q1=q(1:n-nunstab,:);
q2=q(n-nunstab+1:n,:);
z1=z(:,1:n-nunstab)';
z2=z(:,n-nunstab+1:n)';
a2=a(n-nunstab+1:n,n-nunstab+1:n);
b2=b(n-nunstab+1:n,n-nunstab+1:n);
etawt=q2*pi;
% zwt=q2*psi;
% branch below is to handle case of no stable roots, which previously (5/9/09)
% quit with an error in that case.
neta = size(pi,2);
if nunstab == 0
  etawt == zeros(0,neta);
  ueta = zeros(0,0);
  deta = zeros(0,0);
  veta = zeros(neta,0);
  bigev = 0;
else
  [ueta,deta,veta]=svd(etawt);
  md=min(size(deta));
  bigev=find(diag(deta(1:md,1:md))>realsmall);
  ueta=ueta(:,bigev);
  veta=veta(:,bigev);
  deta=deta(bigev,bigev);
end
% ------ corrected code, 3/10/04
eu(1) = length(bigev)>=nunstab;
% ------ Code below allowed "existence" in cases where the initial lagged state was free to take on values
% ------ inconsistent with existence, so long as the state could w.p.1 remain consistent with a stable solution
% ------ if its initial lagged value was consistent with a stable solution.  This is a mistake, though perhaps there
% ------ are situations where we would like to know that this "existence for restricted initial state" situation holds.
%% [uz,dz,vz]=svd(zwt);
%% md=min(size(dz));
%% bigev=find(diag(dz(1:md,1:md))>realsmall);
%% uz=uz(:,bigev);
%% vz=vz(:,bigev);
%% dz=dz(bigev,bigev);
%% if isempty(bigev)
%% 	exist=1;
%% else
%% 	exist=norm(uz-ueta*ueta'*uz) < realsmall*n;
%% end
%% if ~isempty(bigev)
%% 	zwtx0=b2\zwt;
%% 	zwtx=zwtx0;
%% 	M=b2\a2;
%% 	for i=2:nunstab
%% 		zwtx=[M*zwtx zwtx0];
%% 	end
%% 	zwtx=b2*zwtx;
%% 	[ux,dx,vx]=svd(zwtx);
%% 	md=min(size(dx));
%% 	bigev=find(diag(dx(1:md,1:md))>realsmall);
%% 	ux=ux(:,bigev);
%% 	vx=vx(:,bigev);
%% 	dx=dx(bigev,bigev);
%% 	existx=norm(ux-ueta*ueta'*ux) < realsmall*n;
%% else
%% 	existx=1;
%% end
% ----------------------------------------------------
% Note that existence and uniqueness are not just matters of comparing
% numbers of roots and numbers of endogenous errors.  These counts are
% reported below because usually they point to the source of the problem.
% ------------------------------------------------------
% branch below to handle case of no stable roots
if nunstab == n
  etawt1 = zeros(0,neta);
  bigev =0;
  ueta1 = zeros(0, 0);
  veta1 = zeros(neta,0);
  deta1 = zeros(0,0);
else
  etawt1 = q1 * pi;
  ndeta1 = min(n-nunstab,neta);
  [ueta1,deta1,veta1]=svd(etawt1);
  md=min(size(deta1));
  bigev=find(diag(deta1(1:md,1:md))>realsmall);
  ueta1=ueta1(:,bigev);
  veta1=veta1(:,bigev);
  deta1=deta1(bigev,bigev);
end
%% if existx | nunstab==0
%%    %disp('solution exists');
%%    eu(1)=1;
%% else
%%     if exist
%%         %disp('solution exists for unforecastable z only');
%%         eu(1)=-1;
%%     %else
%%         %fprintf(1,'No solution.  %d unstable roots. %d endog errors.\n',nunstab,size(ueta1,2));
%%     end
%%     %disp('Generalized eigenvalues')
%%    %disp(gev);
%%    %md=abs(diag(a))>realsmall;
%%    %ev=diag(md.*diag(a)+(1-md).*diag(b))\ev;
%%    %disp(ev)
%% %   return;
%% end
if isempty(veta1)
	unique=1;
else
	loose = veta1-veta*veta'*veta1;
	[ul,dl,vl] = svd(loose);
	nloose = sum(abs(diag(dl)) > realsmall*n);
	unique = (nloose == 0);
end
if unique
   %disp('solution unique');
   eu(2)=1;
else
   fprintf(1,'Indeterminacy.  %d loose endog errors.\n',nloose);
   %disp('Generalized eigenvalues')
   %disp(gev);
   %md=abs(diag(a))>realsmall;
   %ev=diag(md.*diag(a)+(1-md).*diag(b))\ev;
   %disp(ev)
%   return;
end
tmat = [eye(n-nunstab) -(ueta*(deta\veta')*veta1*deta1*ueta1')'];
G0= [tmat*a; zeros(nunstab,n-nunstab) eye(nunstab)];
G1= [tmat*b; zeros(nunstab,n)];
% ----------------------
% G0 is always non-singular because by construction there are no zeros on
% the diagonal of a(1:n-nunstab,1:n-nunstab), which forms G0's ul corner.
% -----------------------
G0I=inv(G0);
G1=G0I*G1;
usix=n-nunstab+1:n;
C=G0I*[tmat*q*c;(a(usix,usix)-b(usix,usix))\q2*c];
impact=G0I*[tmat*q*psi;zeros(nunstab,size(psi,2))];
fmat=b(usix,usix)\a(usix,usix);
fwt=-b(usix,usix)\q2*psi;
ywt=G0I(:,usix);
% Correction 5/07/2009:  formerly had forgotten to premultiply by G0I
loose = G0I * [etawt1 * (eye(neta) - veta * veta');zeros(nunstab, neta)];
% -------------------- above are output for system in terms of z'y -------
G1=real(z*G1*z');
C=real(z*C);
impact=real(z*impact);
loose = real(z * loose);
% Correction 10/28/96:  formerly line below had real(z*ywt) on rhs, an error.
ywt=z*ywt;
end

% ======================================
% Function: qzdiv
% ======================================

function [A,B,Q,Z,v] = qzdiv(stake,A,B,Q,Z,v)
%function [A,B,Q,Z,v] = qzdiv(stake,A,B,Q,Z,v)
%
% Takes U.T. matrices A, B, orthonormal matrices Q,Z, rearranges them
% so that all cases of abs(B(i,i)/A(i,i))>stake are in lower right 
% corner, while preserving U.T. and orthonormal properties and Q'AZ' and
% Q'BZ'.  The columns of v are sorted correspondingly.
%
% by Christopher A. Sims
% modified (to add v to input and output) 7/27/00
vin = nargin==6;
if ~vin, v=[]; end;
[n jnk] = size(A);
root = abs([diag(A) diag(B)]);
root(:,1) = root(:,1)-(root(:,1)<1.e-13).*(root(:,1)+root(:,2));
root(:,2) = root(:,2)./root(:,1);
for i = n:-1:1
   m=0;
   for j=i:-1:1
      if (root(j,2) > stake | root(j,2) < -.1) 
         m=j;
         break
      end
   end
   if (m==0) 
      return 
   end
   for k=m:1:i-1
      [A B Q Z] = qzswitch(k,A,B,Q,Z);
      tmp = root(k,2);
      root(k,2) = root(k+1,2);
      root(k+1,2) = tmp;
      if vin
         tmp=v(:,k);
         v(:,k)=v(:,k+1);
         v(:,k+1)=tmp;
      end
   end
end         
end

% ======================================
% Function: qzswitch
% ======================================

function [A,B,Q,Z] = qzswitch(i,A,B,Q,Z)
%function [A,B,Q,Z] = qzswitch(i,A,B,Q,Z)
%
% Takes U.T. matrices A, B, orthonormal matrices Q,Z, interchanges
% diagonal elements i and i+1 of both A and B, while maintaining
% Q'AZ' and Q'BZ' unchanged.  If diagonal elements of A and B
% are zero at matching positions, the returned A will have zeros at both
% positions on the diagonal.  This is natural behavior if this routine is used
% to drive all zeros on the diagonal of A to the lower right, but in this case
% the qz transformation is not unique and it is not possible simply to switch
% the positions of the diagonal elements of both A and B.
 realsmall=sqrt(eps)*10;
%realsmall=1e-3;
a = A(i,i); d = B(i,i); b = A(i,i+1); e = B(i,i+1);
c = A(i+1,i+1); f = B(i+1,i+1);
		% A(i:i+1,i:i+1)=[a b; 0 c];
		% B(i:i+1,i:i+1)=[d e; 0 f];
if (abs(c)<realsmall & abs(f)<realsmall)
	if abs(a)<realsmall
		% l.r. coincident 0's with u.l. of A=0; do nothing
		return
	else
		% l.r. coincident zeros; put 0 in u.l. of a
		wz=[b; -a];
		wz=wz/sqrt(wz'*wz);
		wz=[wz [wz(2)';-wz(1)'] ];
		xy=eye(2);
	end
elseif (abs(a)<realsmall & abs(d)<realsmall)
	if abs(c)<realsmall
		% u.l. coincident zeros with l.r. of A=0; do nothing
		return
	else
		% u.l. coincident zeros; put 0 in l.r. of A
		wz=eye(2);
		xy=[c -b];
		xy=xy/sqrt(xy*xy');
		xy=[[xy(2)' -xy(1)'];xy];
	end
else
	% usual case
	wz = [c*e-f*b, (c*d-f*a)'];
	xy = [(b*d-e*a)', (c*d-f*a)'];
	n = sqrt(wz*wz');
	m = sqrt(xy*xy');
	if m<eps*100
		% all elements of A and B proportional
		return
	end
   wz = n\wz;
   xy = m\xy;
   wz = [wz; -wz(2)', wz(1)'];
   xy = [xy;-xy(2)', xy(1)'];
end
A(i:i+1,:) = xy*A(i:i+1,:);
B(i:i+1,:) = xy*B(i:i+1,:);
A(:,i:i+1) = A(:,i:i+1)*wz;
B(:,i:i+1) = B(:,i:i+1)*wz;
Z(:,i:i+1) = Z(:,i:i+1)*wz;
Q(i:i+1,:) = xy*Q(i:i+1,:);
end

% ======================================
% Function: dylap
% ======================================

% This program solves the discrete Lyapunov-equation
%
% Sigma = Gamma*Sigma*Gamma' + Pi*Sigma_{e}*Pi'
%
% for Sigma, i.e. the exact covariance matrix for a VAR(1).
% Sigma_{e} is a covariance matrix for shocks.

function V=dlyap(gamma,pai,sigmae);


sigma0 = eye(length(gamma));
diff = 5;

while diff > 1e-5

sigma1 = gamma*sigma0*gamma' + pai*sigmae*pai';

diff=max(max(abs(sigma1-sigma0)));

sigma0 = sigma1;

end

V = sigma1;
end