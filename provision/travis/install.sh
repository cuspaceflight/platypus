#!/bin/bash
#
# Install software within the Travis CI environment. Requires an Ubuntu Trusty
# worker and sudo access. This script can be viewed additionally as tested
# installation instructions(!)
#
# DO NOT RUN THIS SCRIPT DIRECTLY WITH ROOT PRIVILEGES.

# Log commands and exit on error
set -xe

# Refresh the apt package cache and install the apt repository management
# tooling.
sudo apt-get -y update
sudo apt-get -y install software-properties-common

