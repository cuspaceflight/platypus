import pytest
import numpy as np
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
    
    wv1    = c.calcWorkingVariables(state1,1.4,1005.*(1.-1./1.4))
    
    #Check the computed working variables are correct with zero velocity
    assert wv1[0] == 0.     #Velocity = 0
    assert tolerance(wv1[1],102682.285,1e-3) #Pressure
    assert tolerance(wv1[2],298.,1e-11) #Temperature
    assert tolerance(wv1[3],299490.,1e-11) #Enthalpy
    
    #Check the computed flux is correct (i.e. just the pressure is present)
    f1 = c.calculateFlux(state1,wv1)
    
    assert f1[0] == 0.      # No mass flux
    assert f1[1] == wv1[1]  # Just the pressure term in the momentum equation
    assert f1[2] == 0.      # No energy flux
    