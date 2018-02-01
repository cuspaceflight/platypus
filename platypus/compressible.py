""" Helper functions for compressible flow calculations.

This module contains a series of helper functions for compressbile flow calculations alongisde implemntations
for flux calculations for a simple 1D finite volume code.
"""
import math
import numpy as np

def densityRatioShock(M,gamma):
    """Returns density ratio across a normal shock.
    
    Args:
        M : Mach number of flow entering shock
        gamma: ratio of specific heats
    Returns:
        rho_s/rho
    """
    return ((gamma+1.)*M**2.)/(2.+(gamma-1.)*M**2.)

    
def pressureRatioShock(M,gamma):
    """Returns  static pressure ratio across a normal shock.
    
    Args:
        M : Mach number of flow entering shock
        gamma: ratio of specific heats
    Returns:
        p_s/p
    """
    return 1. + (2.*(gamma))*(M**2.-1.)/(gamma+1.)
def getGasProperties(gasIdent):
    """Returns standard gas properties.
    
    Returns gas properties for some standard gases, assumed to be at a nominal temperature for that gas.
    
    Args:
        gasIdent (str): A string identifying the gas.
        
    Returns:
        [cp, gamma]
    """
    
    if gasIdent == 'air':
        return [1005,1.4]
    elif gasIdent == 'N2O':
        return [880,1.27]
    
def calcStagnTempRatio(gamma,M):
    """Calcuates T/T0 for a given Mach number and gamma.
    
    Args: 
        gamma (float): Ratio of specific heats for the gas
        M (float): Mach number for the gas
    Returns:
        T/T0
    """
    return 1./(1+ 0.5*(gamma-1.)*M*M)

def calcStagnPRatio(gamma,M):
    """Calcuates p/p0 for a given Mach number and gamma.
    
    Args: 
        gamma (float): Ratio of specific heats for the gas
        M (float): Mach number for the gas
    Returns:
        p/p0
    """
    return (1+0.5*(gamma-1)*M*M)**(-gamma/(gamma-1))

def calcLimitMassFlow(gamma):
    """Calcuates the limiting non-dimensional mass flux for a given gamma.
    
    Args: 
        gamma (float): Ratio of specific heats for the gas
    Returns:
        mDot*sqrt(cp*T0)/(A*p0) for the gas at Mach 1
    """
    return (gamma/math.sqrt(gamma-1.))*(1+0.5*(gamma-1))**(-0.5*(gamma+1.)/(gamma-1))

def calcWorkingVariables(state,gamma,R):
    """Calculates the 'working variables' required by a 1D finite volume solver.
    
    Whilst the finite volume solver works by considering conservation and transport of
    the mass per unit volume, the momentum per unit volume and the total energy per unit volume, there
    are a number of quantities it needs to derive from the state variables. These are the local velocity,
    pressure, temperature and enthalpy.
    
    Args:
        state (list of floats): the gas state in the order of rho, rho*u and E_v
        gamma: the ratio of specific heats for the gas
        R: the gas constant for the gas
        
    Returns:
        ([u,p,T,H],maxChicSpeed) a list of the flow velocity, pressure, temperature and enthalpy along with the maximum characteristic speed
    """
    u = state[1]/state[0] # u = rho*u/rho
    p = (state[2]-0.5*state[1]*u)*(gamma-1.)
    T = p/(state[0]*R)
    H = (state[2]+p)/state[0]
    
    maxChicSpeed = max(abs(u+math.sqrt(gamma*R*T)),abs(u-math.sqrt(gamma*R*T)))
    return (np.array([u,p,T,H]),maxChicSpeed)

def calculateFlux(state,wv):
    """Calculates the flux.
    
    This evaluates the standard flux from the given state and working variables.
    
    Args:
        state (list of floats): the gas state in the order of rho, rho*u and E_v
        wv (list of floats): working variables in oder u,p,T,H
    Returns:
        [massFlux,momentumFlux,energyFlux]
    """
    massFlux    = state[1]
    momFlux     = state[1]*wv[0] + wv[1]
    enFlux      = (state[2] + wv[1])*wv[0]
    
    return np.array([ massFlux,momFlux,enFlux])

def calculateSignAFlux(U,a,H,f2,f1,gamma):
    """ Calculates the flux using the sign(A) matrix described in http://www.astro.uu.se/~hoefner/astro/teach/ch10.pdf .
    
    Args:
        U (float): The roe-averaged velocity
        a (float): The roe-averaged speed of sound
        H (float): The roe-averaged enthalpy
        f2 (flux): The flux in the right hand cell using the non roe-averaged state
        f1 (flux): The flux in the left hand cell using the non roe-averaged state
        gamma (float): The ratio of specific heats for the gas
    Returns:
        The flux correction.
    """
    
    signLambda = np.matrix([[np.sign(U-a), 0.       , 0.],
                            [0.          ,np.sign(U), 0.],
                            [0.          ,0.        ,np.sign(U+a)]])
    
    PMat      = np.matrix([[ 1.  ,1.     ,1.],
                           [U-a  ,U      ,U+a],
                           [H-a*U,0.5*U*U,H+a*U]])
    alpha1 = (gamma-1.)*U*U/(2.*a*a)
    alpha2 = (gamma-1.)/(a*a)
    
    PInvMat = np.matrix([[0.5*(alpha1+U/a),-0.5*(alpha2*U + 1./a),0.5*alpha2],
                         [1.-alpha1, alpha2*U, -alpha2],
                         [0.5*(alpha1-U/a),-0.5*(alpha2*U-1./a),0.5*alpha2]])
    
    f2Mat = np.matrix(f2).T
    f1Mat = np.matrix(f1).T
    
    calcFlux =  PMat*signLambda*PInvMat*(f2Mat-f1Mat)
    
    return np.array(calcFlux.T)
                            
def calculateRoeFlux(state1,wv1,state2,wv2,gamma):
    """ Calculates the flux for a Roe's approximate solver scheme.
    
    Args:
        state1 : state in the left hand cell
        wv1    : working variables in the left hand cell
        state2 : state in the right hand cell
        wv2    : working variables in the right hand cell
        gamma  : gamma of the gas
    Returns:
        Flux for the Roe's approximate solver scheme.
    """

    roeAvg = math.sqrt(state2[0]/state1[0])
    
    rhoAvg = roeAvg*state1[0]
    
    UAvg    = (1./(1.+roeAvg))*(roeAvg*wv2[0] + wv1[0])
    
    HAvg    = (1./(1.+roeAvg))*(roeAvg*wv2[3] + wv1[3])
    
    a2Avg   = (gamma-1.)*(HAvg - 0.5*UAvg**2.)
    
    
    flux1 = calculateFlux(state1,wv1)

    
    flux2 = calculateFlux(state2,wv2)

    return 0.5*( flux1 + flux2 - calculateSignAFlux(UAvg,math.sqrt(a2Avg),HAvg,flux2,flux1,gamma))

def reflectionFlux(state,wv):
    """Calculates flux for a reflective wall.
    
    Args:
        state : state in the cell adjacent to the wall
        wv : working variables in the cell adjacent to the wall
    Returns:
        flux (np.array) : the flux for the reflection boundary conditon
    """
    return np.array([0.,wv[1],0.])

class FiniteVolumeSolver1D:
    def __init__(self,cellCentres,areas,states,workingVariables,gasProp,
                 wvFunc = calcWorkingVariables, intercellFluxFunc = calculateRoeFlux,
                 inletFluxFunc = reflectionFlux, outletFluxFunc = reflectionFlux, 
                 source = None, updateFunc = None):
        """Sets up a finite volume solver for a quasi-1D flow geometry.
        
        Args:
            cellCentres (np.array): x-location of the cell centres. The cells are assumed to be equispaced.
            areas (np.array): Area at the cell centre. The area is assumed to vary quadratically (i.e linear scaling of a length scale) between cells
            states (np.array): Initial states at each cell centre
            gasProp [gamma,R]: gas properties
            workingVariables (np.array): Array to store the working variables of each cell
            wvFunc : function to use to calculate any working variables (default: u,p,T,h)
            intercellFluxFunc : function to use to calculate fluxes between cells (default: approximate Roe fluxes)
            inletFluxFunc: function to use to calculate inlet fluxes (default : reflective bc)
            outletFluxFunc : function to use to calculate outlet fluxes (default : reflective bc)
            source: function to calculate internal source terms
            updateFunc : function to perform any additional calculations add the end of the solve
        """
        
        self.cellCentres        = cellCentres
        self.areas              = areas
        self.states             = states
        self.workingVariables   = workingVariables
        self.stateUpdates       = np.zeros(states.shape)
        self.intercellFluxes    = np.zeros((states.shape[0],states.shape[1]+1))
        
        self.gamma  = gasProp[0]
        self.R      = gasProp[1]
        
        #Functions
        self.wvFunc = wvFunc
        self.flux   = intercellFluxFunc
        self.iFlux  = inletFluxFunc
        self.oFlux  = outletFluxFunc
        
        if source == None:
            #We define a blank source
            def zeroSource(state,wv,vol,index):
                return np.zeros(state.shape[0])
            self.source = zeroSource
        else:
            self.source = source
        
        self.updateFunc = updateFunc
        
        self.cellWidth  = cellCentres[1]-cellCentres[0]
        self.nCells     = cellCentres.shape[0]
        
    def calcMidArea(self,area1,area2):
        """Calculates the area at the midpoint between two cells.
        
        It is assumed that the underlying length scale (e.g. radius) varies linearly
        and so the area varies quadratically.
        
        Args:
            area1 (float): area of cell 1
            area2 (float): area of cell 2
        Returns:
            Area at midpoint
        """
        mid     = 0.25*(area1 + area2 + 2.*math.sqrt(area1*area2))
        
        return mid
    
    def calcVolume(self,area1,area2,dist):
        """Calculates the volume between two given planes.
        
        Calculates the volume between the surfaces of given areas with a set distance
        between them. The area is assumed to vary quadratically.
        
        Args:
            area1 (float): first area
            area2 (float): second area
            dist (float): distance between the two surfaces
        """
        
        factor =  (1./dist)*(math.sqrt(area2)/math.sqrt(area1)-1.)
        
        vol = area1*(dist + factor*dist**2. + ((factor**2.)/3.)*dist**3.)
        
        return vol
        
    def timestep(self,CFL,deltaT):
        """Performs a single timestep.
        
        Args:
            CFL (float) : the maximum CFL number to use
            deltaT : the desired timestep to use
        Returns:
            The actual timestep used
        """
        cw = self.cellWidth
        maxChicSpeed = 0.
        
        for i in range(self.nCells):
            #We iterate through the cells and compute the working variables
            (wv,chic) = self.wvFunc(self.states[:,i],self.gamma,self.R)
            
            self.workingVariables[:,i] = wv
            
            if i == 0:
                self.intercellFluxes[:,i] = self.iFlux(self.states[:,i],wv)
            else:
                self.intercellFluxes[:,i] = self.flux(self.states[:,i-1],self.workingVariables[:,i-1],
                                                      self.states[:,i  ],wv,self.gamma)
                
            maxChicSpeed = max(maxChicSpeed,chic)
        #We compute the exit flux
        self.intercellFluxes[:,self.nCells] = self.oFlux(self.states[:,self.nCells-1],
                                                         self.workingVariables[:,self.nCells-1])
        
        #Working variables have been computed so we calculate the delta t using the maximum max characteristic speed

        deltaT = min(deltaT,CFL*self.cellWidth/maxChicSpeed)
        
        #We now compute the fluxes
        for i in range(self.nCells):
            
            if i != self.nCells-1:
                ARhs    = self.calcMidArea(self.areas[i],self.areas[i+1]) #We calculate the area of the flux plane on the right hand side
                volRhs  = self.calcVolume(self.areas[i],ARhs,0.5*cw)
            else:
                ARhs    = self.areas[i]
                volRhs  = 0.5*cw*self.areas[i]
            
            if i != 0:
                ALhs    = self.calcMidArea(self.areas[i-1],self.areas[i]) 
                volLhs  = self.calcVolume(ALhs,self.areas[i],0.5*cw)
            else:
                ALhs    = self.areas[i]
                volLhs  = 0.5*cw*self.areas[i]
                
            vol = volLhs + volRhs   #We construct the volume
            source = self.source(self.states[:,i],self.workingVariables[:,i],vol,i) # We apply any source term
            
            self.stateUpdates[:,i]  = (1./vol)*( ALhs*self.intercellFluxes[:,i]     #We calculate the state update
                                                -ARhs*self.intercellFluxes[:,i+1]
                                                +source)
            
        self.states += self.stateUpdates*deltaT
        
        #We call an update function
        if self.updateFunc != None:
            self.updateFunc(self)   #This allows the update function to do whatever it wants 
        
        return deltaT
            
            
            
                
                
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    