mltools
=======

## Code-Collection for Machine-Learning applications

### The idea behind mltools
The mltools-package represents a collection of tools for applications in the field of machine-learning.

### Contribute
If you want to contribute to mltools, we propose the same workflow as also suggested for [ase](https://wiki.fysik.dtu.dk/ase/development/contribute.html#).
In general this means

* Never work in master branch locally or on GitLab.
* Make a new branch for whatever you are doing. When you are done, push it to your own repository and make a merge request from that branch in your repository to official master.


As it comes to commit messages, please follow [numpy's guidelines](http://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html)
which in particular require to start commit messages with an appropriate acronym:

```
API: an (incompatible) API change
BLD: change related to building numpy
BUG: bug fix
DEP: deprecate something, or remove a deprecated object
DEV: development tool or utility
DOC: documentation
ENH: enhancement
MAINT: maintenance commit (refactoring, typos, etc.)
REV: revert an earlier commit
STY: style fix (whitespace, PEP8)
TST: addition or modification of tests
```
### Requirements
Required for mltools are:

* ase     >= 3.18.0b1
* nose    >= 1.3.7
* numpy   >= 1.16.1
* scipy   >= 1.2.1
* pandas  >= 0.23.1
