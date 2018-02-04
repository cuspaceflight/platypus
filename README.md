Platypus
==========

[![Build Status](https://travis-ci.org/cuspaceflight/platypus.svg?branch=master)](https://travis-ci.org/cuspaceflight/platypus)
[![Coverage Status](https://coveralls.io/repos/github/cuspaceflight/platypus/badge.svg?branch=master)](https://coveralls.io/github/cuspaceflight/platypus?branch=master)


The aim of platypus is to develop a robust simulation tool for hybrid rocket motors that aims to 
capture some of the unsteady dynamics within the motor. This should allow greater insight into potential
transients during motor start-up as well and potential instabilities of the motor.

# Installation and usage

At the moment the project consists of helper functions for a 1D finite volume solver using Roe's approximate
fluxes.

To install use:

```
pip install git+https://github.com/cuspaceflight/platypus.git
```

# Validation

To perform validation of the solver a validation suite is present and can be run through:

```
tox
```

This tests the various flow relationships implemented in the helper functions as well as 
validating the 1D finite volume solver against the analytic solutions of Sod's shock tube and
a flow through a nozzle.

# Project goals

The current simulation tool used by CUSF for motor simulation uses a quasi-steady process, treating the
fuel grain as a single unit. At each step the oxidiser mass flow is calculated by using the tank pressure
and current chamber pressure, the corresponding fuel flux is found, the gas properties based on the O/F ratio
are computed and then the mass flow through the nozzle is calculated. An iterative scheme is then applied
to adjust the chamber pressure so that these mass flows are equal. The tank pressure is then updated based
on the oxidiser mass flow and time step and the process is repeated.

Platypus aims to investigate the unsteady dynamics that the above method masks. By investigating
variation of regression rate and gas composition along the motor length then the effect of the varying 
O/F ratio on motor performance will be revealed.

In addition to developing a robust simulation code the other goal of this project is to provide decent documentation
such that future CUSF projects and other groups can make use of and improve these tools.

