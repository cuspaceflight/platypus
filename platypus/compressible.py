""" Helper functions for compressible flow calculations.

This module contains a series of helper functions for compressbile flow calculations alongisde implemntations
for flux calculations for a simple 1D finite volume code.
"""
import math
import numpy as np


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
        [u,p,T,H] a list of the flow velocity, pressure, temperature and enthalpy.
    """
    u = state[1]/state[0] # u = rho*u/rho
    p = (state[2]-0.5*state[1]*u)*(gamma-1.)
    T = p/(state[0]*R)
    H = (state[2]+p)/state[0]
    
    return [u,p,T,H]

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

    