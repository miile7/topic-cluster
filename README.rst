.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=============
topic-cluster
=============


    Cluster papers into topics according to their titles and abstracts.


This program takes a Bibtex file and reads the titles and/or the abstracts. After that a Latent Dirichlet Allocation is applied on those. The result is shown as a bar graph. The idea of this is found in Teh et. al. [1]_

Installation
============

Manual installation
-------------------

1. Download the repository
2. Run ``pip install .``
3. Done

Usage
=====

.. image:: docs/screenshot.jpg



The program can either be run as a pure command line program. For this run ``topic_cluster``. The following arguments are supported:

.. code-block:: bash

    usage: topic_cluster [-h] [--version] [-v] [-vv] [-t TOPICS] [-f FEATURES] [--no-title] [--no-abstract] [bibtex_path]

**Positional optional arguments**

- ``bibtex_path``: The file path of the bibtex file to read

**Optional arguments**

- ``-h``, ``--help``: Show this help message and exit
- ``--version``: Show program's version number and exit
- ``-v``, ``--verbose``: Set loglevel to INFO
- ``-vv``, ``--very-verbose``: Set loglevel to DEBUG
- ``-t TOPICS``, ``--topics TOPICS``: Set the number of topics, default is 3
- ``-f FEATURES``, ``--features FEATURES``: Set the number of features to for each topic, default is 10
- ``--no-title``: Use to exclude the title from the feature detection
- ``--no-abstract``: Use to exclude the abstract from the feature detection

If no ``bibtex_path`` is given, a dialog will ask for the bibtex path. The topic and feature count will have the default values and title and abstract are used.y

The actual appearance of the graph depends on the backend used by matplotlib.

To use natural language processing to refine search terms is called "systematic reviews" which I found in Teh et. al. [1]_


Note
====

This project has been set up using PyScaffold 4.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.

.. [1] Teh, Hui Yie, Andreas W. Kempa-Liehr, und Kevin I-Kai Wang. "Sensor data quality: a systematic review". Journal of Big Data 7, Nr. 1 (11. Februar 2020): 11. https://doi.org/10.1186/s40537-020-0285-1.