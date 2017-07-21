# PAM Summer Intern Project 2017

Code for testing CNN classification of blue and fin whales

## Prerequisites

- Raven software TBD version
- Python version  3.6.1, on a Mac OSX download and install from here:
 https://www.python.org/downloads/mac-osx/


## Running

Create virtual environment with correct dependencies

    $ pip3 install virtualenv
    $ virtualenv --python=/usr/bin/python3.6 venv-pam
    $ source venv-pam/bin/activate
    $ pip3 install -r requirements.txt
    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py3-none-any.whl
    $ pip3 install --upgrade $TF_BINARY_URL

Check-out code

    $ git clone https://github.com/daniel-deleon/PAM

TODO: Add steps for running below

*  Run BLED detector
*  Save results to folder in TBD format
*  etc.

## Developer Notes

A placeholder for notes that might be useful for developers

