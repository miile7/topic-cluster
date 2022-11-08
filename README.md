# topic-cluster

> Cluster papers into topics according to their titles and abstracts.

This program takes a Bibtex file and reads the titles and/or the
abstracts. After that a Latent Dirichlet Allocation is applied on those.
The result is shown as a bar graph. The idea of this is found in Teh et.
al.[^1]

![image](docs/screenshot.jpg)

## Installation

To run this program from the code directly, [`python`](https://www.python.org/) and [`poetry`](https://python-poetry.org/) (`pip install poetry`) are required.

To install all the dependencies, use your command line and navigate to the directory where this `README` file is located in. Then run

```bash
poetry install
```

## Execution

To execute the program use
```bash
poetry run python -m topic_cluster
```

The following arguments are supported:

``` bash
topic_cluster [-h] [--version] [-v] [-vv] [-t TOPICS] [-f FEATURES] [--no-title] [--no-abstract] [bibtex_path]
```

**Positional optional arguments**

  - `bibtex_path`: The file path of the bibtex file to read

**Optional arguments**

  - `-h`, `--help`: Show this help message and exit
  - `--version`: Show program's version number and exit
  - `-v`, `--verbose`: Set loglevel to INFO
  - `-vv`, `--very-verbose`: Set loglevel to DEBUG
  - `-t TOPICS`, `--topics TOPICS`: Set the number of topics, default is
    3
  - `-f FEATURES`, `--features FEATURES`: Set the number of features to
    for each topic, default is 10
  - `--no-title`: Use to exclude the title from the feature detection
  - `--no-abstract`: Use to exclude the abstract from the feature
    detection

If no `bibtex_path` is given, a dialog will ask for the bibtex path. The
topic and feature count will have the default values and title and
abstract are used.

The actual appearance of the graph depends on the backend used by
matplotlib.

To use natural language processing to refine search terms is called
"systematic reviews" which I found in Teh et. al.[^1]

[^1]: Teh, Hui Yie, Andreas W. Kempa-Liehr, und Kevin I-Kai Wang.
    "Sensor data quality: a systematic review". Journal of Big Data 7,
    Nr. 1 (11. Februar 2020): 11. <https://doi.org/10.1186/s40537-020-0285-1>.
