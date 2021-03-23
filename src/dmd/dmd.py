#! /usr/bin/env python

import numpy as np
import pickle


class DMDBase(object):
    """
    I'm just going to work through the basics of Dynamic Mode Decomposition

    Start from a dynamical system
    \begin{eqnarray}
    \label{eq:ode-full}
    \od{\vec x}{t} &=& \vec f(\vec x,t;\mu)
    \end{eqnarray}
    Discretize in time, fix $\mu$ and create a flow map
    \begin{eqnarray}
    \vec x_{k+1} &=& \vec F(\vec x_k)
    \end{eqnarray}
    where $\vec x_{k}$ is an approximation to $\vec x$ at $t_k=t_0+k\Delta t$. We also have measurements explicitly denoted as
    \begin{eqnarray}
    \vec y_k &=& \vec g(\vec x_k)
    \end{eqnarray}

    The idea is to build a linear approximate $\mathcal{A}$ to $\vec f(\vec x,t)$
    \begin{eqnarray}
    \label{eq:ode-linA-cont}
    \od{\vec x}{t} &=&\mathcal{A}\vec x
    \end{eqnarray}
    and then address \eqn{ode-linA-cont} rigorously since it's linear.

    In fact, the solution to \eqn{ode-linA-cont} is
    \begin{eqnarray}
    \vec x(t) = \sum_{k=1}^{n}\gvec \phi_k\exp(\omega_k t)b_k = \gvec\Phi\exp(\Omega t)\vec b
    \end{eqnarray}
    where the pairs $\gvec \phi_k,\omega_k$ are the eigenvectors and values for $\mathcal{A}$ and $b_k$ are the components of $\vec x(0)$ in the eigenbasis.

    In the discrete setting, we have the matrix $\mat A = \exp(\mathcal{A}\Delta t)$ and
    \begin{eqnarray}
    \label{eq:ode-linA-disc}
    \vec x_{k+1} &=& \mat A\vec x_{k}
    \end{eqnarray}
    The solution to \eqn{ode-linA-disc} has an analogous solution
    \begin{eqnarray}
    \vec x_{k+1}=\sum_{j=1}^n \gvec \phi_j\lambda^k_jb_j = \gvec \Phi\gvec \Lambda^k\vec b
    \end{eqnarray}
    Where again $\vec \phi_j$ and $\lambda_j$ are the eigenvectors and eigenvalues of $\mat A$ and $\vec b=\gvec \Phi \vec x_1$.

    The idea is that the DMD produces a low-rank decomposition of $\mat A$ that optimizes the fit of the trajectories $\vec x_{1},\vec x_2,\dots,\vec x_m$ so that
    \begin{eqnarray}
    \|\vec x_{k+1}-\mat A\vec x_k\|_2
    \end{eqnarray}
    is minimized in a least square sense for points $\vec x_{1},\dots,\vec x_{m-1}$.

    The approach is very similar to what we are doing with the NIROM approach, if we start to look at the discrete flow map. We construct two snapshot matrices in $\Re^{N\times m-1}$
    \begin{eqnarray}
    \mat X &=& \lrb{\vec x_1 | \vec x_2 | \dots | \vec x_{m-1}}, \; \mbox{ and } \\
    \mat X^{\prime} &=& \lrb{\vec x_2 | \vec x_3 | \dots | \vec x_{m}}, \\
    \end{eqnarray}
    which then satisfy
    \begin{eqnarray}
    \mat{X^{\prime}}&\approx&\mat{A}\mat{X}
    \end{eqnarray}
    Of course, $\mat X$ and $\mat X^{\prime}$ are not square. They need in fact to be skinny ($N > m$). We can then solve for $\mat{A}$ in a least-squares sense.
    \begin{eqnarray}
    \mat{A} &=& \mat{X^{\prime}}\mat{X}^{\dagger}
    \end{eqnarray}
    where $\mat{X}^{\dagger}$ is the Moore-Penrose pseudoinverse
    \begin{eqnarray}
    \mat{X}^{\dagger}&=&\lrp{\mat X^{T}\mat{X}}^{-1}\mat{X}^T
    \end{eqnarray}
    This minimizes $\left\|\mat X^{\prime}-\mat{A}\mat{X}\right\|_F$ in the Frobenius norm. In fact, performing this in the full dimension of $\mat X$ would be prohibitive for most systems, so the DMD approach includes a fundamental truncation step as well to use a low-rank approximation $\widetilde{\mat A}$

    ## Quick note on the Koopman operator
    The Koopman operator is a linear operator that represents the dynamical systems action on the space of scalar 'measurement' functions (functionals). Specifically, if we define a measurement function $g:\Ce^{n}\rightarrow \Ce$, then the Koopman operator acting on $g$ is just the composition of $g$ with the flow map $\vec F$ for the dynamical system
    \begin{eqnarray}
    \mathcal{K}g=g\circ \vec F \\
    \mathcal{K}g(\vec x_{k}) = g(\vec F(\vec x_k)) = g(\vec x_{k+1})
    \label{eq:Koopman-def}
    \end{eqnarray}
    ## The DMD algorithm
    Given a dynamical system \eqn{ode-full}, and a matrix of snapshots $\mat X$ taken with step $\Delta t$
    \begin{eqnarray}
    \mat X &=& \lrb{\vec x_1 | \vec x_2 | \dots | \vec x_{m-1}}, \; \mat{X} \in \Re^{N\times m-1} \\
    \end{eqnarray}
    The basic DMD algorithm is
    \begin{enumerate}
    \item Take the SVD of $\mat X$ and truncate it to a level $r$
    \begin{eqnarray}
    \mat{X} &\approx& \mat{U}_r\gvec{\Sigma}_r\mat{V}_r^{*} \\
    &&\mat{U}_r \in \Ce^{N\times r}, \; \gvec{\Sigma}_r \in \Ce^{r\times r}, \mbox{ and } \mat{V}\in \Ce^{m-1\times r}
    \end{eqnarray}
    For simplicity we'll drop the $r$ subscript in the following
    \item The matrix $\mat{A}$ can be defined as (recall $\mat{A}=\mat{X}^{\prime}\mat{X}^{\dagger}$)
    \begin{eqnarray}
    \label{eq:A-def}
    \mat{A} &=&\mat{X}^{\prime}V\gvec{\Sigma}^{-1}\mat{U}^{*}
    \end{eqnarray}
    or according to Kuntz et al, it is more efficient to compute the projection of $\mat{A}$ onto the POD modes
    \begin{eqnarray}
    \label{eq:Atilde-def}
    \tilde{\mat{A}}=\mat{U}^{*}\mat{A}\mat{U} = \mat{U}^{*}\mat{X}^{\prime}\mat{V}\gvec{\Sigma}^{-1}
    \end{eqnarray}
    $\tilde{\mat{A}}$ defines a low ($r$) dimensional linear model of the dynamical system on POD coordinates
    \begin{eqnarray}
    \tilde{\vec x}_{k+1}=\tilde{\mat{A}}\tilde{\vec x}_{k}, \; \mbox{ with }
    \vec x_{k} &=& \mat{U}\tilde{\vec x}_{k}
    \end{eqnarray}
    \item Next we look at the temporal dynamics of the linear operator $\tilde{\mat{A}}$. That is we compute the eigen decomposition of the $r\times r$ matrix
    \begin{eqnarray}
    \label{eq:Aeig-def}
    \tilde{\mat{A}}\mat{W} &=& \mat{W}\gvec{\Lambda}
    \end{eqnarray}
    where $\gvec{\Lambda}$ is diagonal and contains the eigenvalues for $\tilde{\mat{A}}$. To recover the eigendecomposition for $\mat{A}$. The matrix (columns) of eigenvectors for $\mat{A}$ are given by
    \begin{eqnarray}
    \label{eq:Phi-def-1}
    \gvec{\Phi} &=&\mat{X}^{\prime}\mat{V}\gvec{\Sigma}^{-1}\mat{W}
    \end{eqnarray}
    \Eqn{Phi-def-1} is apparently referred to as the exact DMD mode decomposition. An alternative definition that needs to be used if $\lambda_k = 0$ is
    \begin{eqnarray}
    \label{eq:Phi-def-2}
    \gvec{\Phi} &=& \mat{U}\mat{W}
    \end{eqnarray}
    \item To recover $\vec x(t)$ we then set $\omega_k=\ln(\lambda_k)/\Delta t$
    \begin{eqnarray}
    \label{eq:xt-eval-1}
    \vec{x}(t) &\approx &\sum_{k=1}^r\gvec{\phi}_k\exp(\omega_k t)b_k = \gvec{\Phi}\exp(\gvec{\Omega}t)\vec b, \; \mbox{ with } \\
    \vec b &=&\gvec{\Phi}^{\dagger}\vec {x}_1
    \end{eqnarray}
    \end{enumerate}

    """
    @staticmethod
    def save_to_disk(filename,ROM,**options):
        """
        Save the instance in ROM to disk using options
        Start with absolutely simplest approach using pickle
        """
        outfile = open(filename,'wb')
        protocol= options.get('protocol',pickle.HIGHEST_PROTOCOL)
        try:
            pickle.dump(ROM,outfile,protocol=protocol)
        finally:
            outfile.close()
        return ROM
    @staticmethod
    def read_from_disk(filename,**options):
        """
        Read the instance in ROM to disk using options
        Start with absolutely simplest approach using pickle
        """
        infile = open(filename,'rb')
        encoding= options.get('encoding','latin1')
        ROM = None
        try:
            ROM = pickle.load(infile,encoding=encoding)
        except TypeError:
            ROM = pickle.load(infile)
        finally:
            infile.close()
        return ROM

    def __init__(self,rank,**options):
        self._rank=rank
        for thing in ['_U_r','_Sigma_r','_V_r','_X']:
            setattr(self,thing,None)
    @property
    def rank(self):
        """
        Truncation level
        """
        return self._rank
    @property
    def n_snap(self):
        """
        return number of snapshots used in training
        """
        return self._n_snap_fit
    @property
    def n_fine(self):
        """
        return fine grid dimension
        """
        return self._Phi.shape[0]
    @property
    def t0_fit(self):
        """
        initial time value for DMD calculation
        """
        return self._t0_fit
    @property
    def dt_fit(self):
        """
        time step assumed for fit
        """
        return self._dt_fit
    @property
    def approximate_operator(self):
        """
        Best fit operator truncated to mode dim
        """
        return self._Atilde
    @property
    def basis(self):
        """
        DMD basis (truncated)
        """
        return self._Phi
    @property
    def omega(self):
        return self._omega
    @property
    def amplitudes(self):
        return self._b
    @property
    def pod_basis(self):
        return self._U_r
    @property
    def pod_singular_values(self):
        return self._Sigma_r

    def fit_basis(self,S,dt_fit,t0_fit=0.):
        """
        Vanilla DMD algorithm
        """
        X,Xp=S[:,0:-1],S[:,1:]
        #time information
        self._t0_fit,self._dt_fit=t0_fit,dt_fit
        t0,dt=0.,1.
        ## compute svd
        U,Sigma,Vt=np.linalg.svd(X,full_matrices=False)
        V=(Vt.conj()).transpose()
        r = min(self._rank,U.shape[1])
        U_r,Sigma_r,V_r=U[:,:r],Sigma[:r],V[:,:r]
        #invert Sigma (diagonal)
        assert np.abs(Sigma_r).min()>0.
        SigmaInv_r=np.reciprocal(Sigma_r)
        SigmaInv_r=np.diag(SigmaInv_r)

        ## compute approximate temporal dynamics through Atilde
        tmp     = ((U_r.conj().transpose()).dot(Xp)).dot(V_r)
        Atilde = tmp.dot(SigmaInv_r)
        # discrete time eigenvalues and eigenvectors
        D,W    = np.linalg.eig(Atilde)
        ## DMD modes
        tmp    = (Xp.dot(V_r)).dot(SigmaInv_r)
        Phi    = tmp.dot(W)
        ## continuous time eigenvalues
        omega  = np.log(D)/dt
        assert np.isnan(omega).any() == False

        ## amplitudes from projection of initial condition
        x0=X[:,0]
        PhiInv=np.linalg.pinv(Phi)
        b=PhiInv.dot(x0)

        mm1 = X.shape[1]

        toffline=t0+np.arange(0,mm1)*dt
        time_dynamics = np.zeros((r,mm1),dtype=np.complex_)

        for kk in range(mm1):
            time_dynamics[:,kk] = b*(np.exp(omega*toffline[kk]))
        #
        Xdmd = Phi.dot(time_dynamics)
        Xdmd = Xdmd.real

        ##keep around
        #POD modes
        self._n_snap_fit=S.shape[1]
        self._U_r,self._Sigma_r,self._SigmaInv_r,self._V_r=U_r,Sigma_r,SigmaInv_r,V_r
        #approximate linear system operator
        self._Atilde=Atilde
        #DMD modes,frequencies, eignenvectors, and projection of the initial condition
        self._Phi,self._omega,self._D,self._b = Phi,omega,D,b
        self._PhiInv = PhiInv

        return Phi,omega,D,b,Xdmd,time_dynamics,U,Sigma,V

    def predict(self,t):

        tnorm=(t-self.t0_fit)/(self.dt_fit)
        time_dynamics= self.amplitudes*(np.exp(self.omega*tnorm))
        uh= self.basis.dot(time_dynamics).real

        return uh
