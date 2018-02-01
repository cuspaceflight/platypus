import pytest
import numpy as np
import math
from platypus import compressible as c

"""Test cases for the compressible relations in compressible.py"""

def tolerance(computedValue,targetValue,tolerance):
    """Checks that the computed value is equal to the target, known value within a certain tolerance"""

    if abs(computedValue-targetValue)<tolerance:
        return True
    return False
    
def test_1dRelations():
    """We test the 1D compressible relations to those results found in the CUED flow tables"""
    
    #Dimensionless mass fluxes at choking
    assert tolerance(c.calcLimitMassFlow(1.4),1.281,1e-4)
    assert tolerance(c.calcLimitMassFlow(1.333),1.3468,1e-4)
    
    #Stagnation temperature and pressure ratios
    assert tolerance(c.calcStagnTempRatio(1.333,0.3),0.9852,1e-4)
    assert tolerance(c.calcStagnTempRatio(1.4,1.260),0.7590,1e-4)
    
    assert tolerance(c.calcStagnPRatio(1.333,1.21),0.4176,1e-4)
    assert tolerance(c.calcStagnPRatio(1.4,0.86),0.6170,1e-4)
    
def test_fluxes():
    """Tests flux computations and calculation of working variables from the state"""
    #We define two test cases for different flow states
    state1 = np.array([1.2,0.,298*(1005./1.4)*1.2]) #Rho=1.2,u=0,T=298 and the gas is air
    
    (wv1,chic1)    = c.calcWorkingVariables(state1,1.4,1005.*(1.-1./1.4))
    
    #Check the computed working variables are correct with zero velocity
    assert wv1[0] == 0.     #Velocity = 0
    assert tolerance(wv1[1],102682.285,1e-3) #Pressure
    assert tolerance(wv1[2],298.,1e-11) #Temperature
    assert tolerance(wv1[3],299490.,1e-11) #Enthalpy
    assert tolerance(chic1,math.sqrt(1.4*298*1005.*(1-1./1.4)),1e-10) #Speed of sound
    
    
    #Check the computed flux is correct (i.e. just the pressure is present)
    f1 = c.calculateFlux(state1,wv1)
    
    assert f1[0] == 0.      # No mass flux
    assert f1[1] == wv1[1]  # Just the pressure term in the momentum equation
    assert f1[2] == 0.      # No energy flux
    

def test_shock_relations():
    #We compare this against tabulated values for M=1.76 at gamma-1.4
    pRatio  = c.pressureRatioShock(1.76,1.4)
    assert tolerance(pRatio,3.4472,1e-4)
    
    #The temperature ratio is tabulated but not the density ratio
    TRatio = 1.5019
    densityRatioAct = 3.4472/TRatio
    
    assert tolerance(c.densityRatioShock(1.76,1.4),densityRatioAct,1e-4)
    
def test_solver_shockTube():
    """Tests the finite volume solver via Sod's shock tube.
    
    It is worth drawing a brief diagram of what the flow should look like. At t =0 there is
    a diaphragm in the middle of a tube separating a high pressure region on the left from
    a low pressure region on the right.
    
    At t = 0 this diaphragm bursts. Some time later (but before reflections occur!) we have:
    
    -----------------------------------------------
    | L      | E          |  2      |   1    | R   |
    -----------------------------------------------
    
    L and R are regions where the gas has yet to be perturbed as the characteristics have not yet reached here.
    
    Region E is an expansion fan.
    Between 1 and 2 exists a contact discontinuity where the density and temperature become discontinuous.
    Between 1 and R exists a shock wave travelling to the right.
    
    """
    
    nCells = 400        #400 cells
    cw     = 1./nCells  #Cell width
    cellCentres     = np.linspace(cw/2.,1.-cw/2.,nCells)    #Calculate cell centres
    cellAreas       = np.ones(nCells)
    states          = np.zeros([3,nCells])
    wv              = np.zeros([4,nCells])
    
    gamma = 1.4
    R     = 1.
    
    solver = c.FiniteVolumeSolver1D(cellCentres,cellAreas,states,wv,[gamma,R]) #Define solver with default fluxes and reflective boundaries
    
    assert solver.calcMidArea(cellAreas[1],cellAreas[2]) == 1. #Basic check of solver getting areas right
    assert solver.calcVolume(cellAreas[1],cellAreas[2],cw) == cw
    assert tolerance(solver.calcVolume(1.,2.,1.),(1./3.)*(3.+math.sqrt(2)),1e-10)  #Volume of a a frustrum with height 1, r= (1./sqrt{pi}), R = sqrt(2/pi)
    
    
    #We calculate the analytic solution
    pL      = 1.
    pR      = 0.1
    rhoL    = 1.0
    rhoR    = 0.125
    aL      = math.sqrt(gamma*pL/rhoL)
    aR      = math.sqrt(gamma*pR/rhoR)
    
    def resid(Ms):
        resid = Ms - 1./Ms - aL*((gamma+1.)/(gamma-1.))*(1.-((pR/pL)*(Ms*Ms*(2.*gamma)/(gamma+1.) - (gamma-1.)/(gamma+1.)))**((gamma-1.)/(2.*gamma)))    
        return resid
    
    def jacob(Ms):
        jacob = 1. + 1./(Ms*Ms) + aL*((gamma+1.)/(gamma-1.))*((gamma-1.)/(2.*gamma))*2.*gamma/(gamma+1.)*2.*Ms*(pR/pL)*((pR/pL)*(Ms*Ms*(2.*gamma)/(gamma+1.) - (gamma-1.)/(gamma+1.)))**((gamma-1.)/(2.*gamma)-1.)
        return jacob
    
    guessMs  = 1.5
    
    tol = 1e-8
    
    #We use a newton method to find the speed of the shock wave
    while abs(resid(guessMs)) > tol:
        deltaMs = - resid(guessMs)/jacob(guessMs)
        
        guessMs += deltaMs
        
    Ms = guessMs
    
    #We now calculate the analytical solution with this knowledge
    def calcAnalyticalSolution(t):
        soln = np.zeros([3,nCells])
        
        Us  = Ms*aR     # The speed of the shock in the lab referenc frame
        
        rho1 = rhoR*c.densityRatioShock(Ms,gamma) #The density ratio across the shock
        U1   = Us*(1.-rhoR/rho1)                  #Velocity after the shock (continuity) 
        p1   = pR*c.pressureRatioShock(Ms,gamma)  #The pressure after the shock
        
        #We relate quantities from region 1 to region 2 
        U2   = U1
        p2   = p1
        rho2 = rhoL*((p2/pL)**(1./gamma))
        a2   = aL-U2*(gamma-1.)/2.
        
        #We calculate the abscissas of the various zones
        xShock      = 0.5+t*Us      #Location of the shock. Everything to the right of this should be undisturbed
        xContact    = 0.5+t*U1      #Location of the contact surface. The density is discontinuous at this point
        xExpansionL = 0.5 - t*aL    #Location of the 'first' ch'ic to leave the discontinuity at t=0
        xExpansionR = 0.5 +(U2-a2)*t#Location of the 'last'  ch'ic to leave the discontinuity at t=0 
        
        print(xShock)
        print(xContact)
        print(xExpansionL)
        print(xExpansionR)
        #we can now work out the analytic state
        for i in range(nCells):
            xLoc = cellCentres[i]
            
            if xLoc < xExpansionL:
                #Region L
                u   = 0.
                rho = rhoL
                p   = pL
            elif xLoc < xExpansionR:
                #Region E
                u   = (2./(gamma+1.))*(aL + (xLoc-0.5)/t)
                a   = aL - (gamma-1.)*u/2.
                p   = pL*(a/aL)**(2.*gamma/(gamma-1.))
                rho = gamma*p/(a**2.)
            elif xLoc<xContact:
                #Region 2
                u   = U2
                rho = rho2
                p   = p2
            elif xLoc<xShock:
                #Region 1
                u   = U1
                rho = rho1
                p   = p1
            else:
                #region R
                u   = 0
                p   = pR
                rho = rhoR
            
            soln[0,i] = rho
            soln[1,i] = rho*u
            soln[2,i] = p/(gamma-1.) + 0.5*rho*u*u
        return soln
        
    
    #We set up the initial conditions
    for i in range(nCells):
        if i < nCells/2:
            states[0,i] = 1.0
            p = 1.
        else:
            states[0,i] = 0.125
            p =0.1
        
        states[2,i] = p/(gamma-1)
    
    #We time march
    t = 0.
    targT = 0.2*(1./aR) #Target non dimensional t
    
    while t < targT:
        deltaT = targT - t #Desired timestep
        
        deltaT = solver.timestep(0.5,deltaT)
        
        t += deltaT
        
    analyticSolution = calcAnalyticalSolution(t)
    
    cumRhoError = 0.
    cumMomError = 0.
    cumEnError  = 0.
    

    for i in range(nCells):
        xLoc = cellCentres[i]

        errRho = abs(states[0,i]-analyticSolution[0,i])/abs(analyticSolution[0,i])
        errMom = abs(states[1,i]-analyticSolution[1,i]) #We can't normalise as velocity goes to zero in somer regions
        errEn  = abs(states[2,i]-analyticSolution[2,i])/abs(analyticSolution[2,i])
        
        cumRhoError += errRho*cw
        cumMomError += errMom*cw
        cumEnError  += errEn *cw

    tol = 5e-2 #A 5% error is pretty good. Roe's approxmiate solver is known to smooth out the expansion fan and contact dicontinuity a fair bit
    assert cumRhoError  < tol
    assert cumMomError  < tol
    assert cumEnError   < tol
    