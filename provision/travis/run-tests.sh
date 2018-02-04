#!/bin/bash
#
# Run tests on Travis CI worker. Requires that travis-install.sh has been run.

# Exit on error
set -e


# Now switch on logging so that we don't get log spam from the OpenFOAM setup
set -x

# Run tox
tox