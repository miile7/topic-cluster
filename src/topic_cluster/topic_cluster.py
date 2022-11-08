from argparse import ArgumentParser, Namespace
from bibtexparser import load  # ignore [import]
from collections import defaultdict
from easygui import fileopenbox
from easysettings import EasySettings
from logging import DEBUG, INFO, getLogger
from numpy.typing import NDArray
from os import path
from pathlib import Path
from re import compile
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tempfile import gettempdir
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
    cast,
)
from typing_extensions import Unpack

import matplotlib.pyplot as plt
import numpy as np

from topic_cluster.version import get_version

NAME = "topic_cluster"


NON_ASCII = compile(r"[\W]+")
CONFIG_PATH = path.join(gettempdir(), "topic_cluster.conf")
CONFIG_OPEN_PATH_INDEX = "bibtex_open_path"
DEFAULT_NUMBER_OF_TOPICS = 3
DEFAULT_NUMBER_OF_FEATURES = 7

logger = getLogger("topic_cluster")


class ParserArgs(TypedDict, total=False):
    bibtex_path: Optional[str]
    no_title: bool
    no_abstract: bool
    feature_count: int
    topic_count: int


class Model(Protocol):
    components_: Iterable[NDArray[np.float64]]
    # components_: ArrayLike


StackedIterable = Union[Any, Iterable["StackedIterable"]]


def get_last_open_file() -> Optional[str]:
    logger.debug(f"Trying to find config file {CONFIG_PATH} for last open path")
    if not path.exists(CONFIG_PATH):
        logger.debug("File does not exist")
        return None

    settings = EasySettings(CONFIG_PATH)

    open_path = settings.get(CONFIG_OPEN_PATH_INDEX)
    logger.debug(f"Found last open path to be {open_path}")
    return open_path


def save_last_open_file(open_path: str) -> None:
    logger.debug(f"Saving last open path {open_path} to config file {CONFIG_PATH}")
    settings = EasySettings(CONFIG_PATH)
    settings.set(CONFIG_OPEN_PATH_INDEX, open_path)
    settings.save()


def get_bibtex_path_by_file_open() -> Optional[str]:
    open_path = get_last_open_file()
    logger.debug("Showing file open dialog")

    open_path = cast(
        Optional[str],
        fileopenbox(
            "No bibtex file found",
            "Select bibtex file",
            open_path or "*",
            [[".bibtex", ".bib", "Bibtex files"]],
            multiple=False,
        ),
    )

    if open_path is None:
        return None

    logger.debug(f"Selected file {open_path}")

    save_last_open_file(open_path)
    return open_path


def get_documents(
    bibtex_path: Optional[str] = None, title: bool = True, abstract: bool = True
) -> List[str]:
    logger.info("Creating documents")

    open_path = (
        bibtex_path if bibtex_path is not None else get_bibtex_path_by_file_open()
    )

    if open_path is None:
        raise ValueError("No bibtex file is given.")

    with open(open_path, mode="r", encoding="utf-8") as bibtex_file:
        db = load(bibtex_file)

    texts = []
    keys = (["title"] if title else []) + (["abstract"] if abstract else [])
    sources: Dict[str, int] = defaultdict(int)

    for entry in db.entries:
        text = []
        for key in keys:
            if key not in entry:
                logger.warning(
                    f"The entry {entry['ID']} ({entry['ENTRYTYPE']}) has no {key}"
                )
            else:
                sources[key] += 1
                logger.debug(
                    f"Found {key} for entry {entry['ID']} ({entry['ENTRYTYPE']})"
                )
                text.append(entry[key])

        if len(text) == 0:
            logger.warning(
                f"The entry {entry['ID']} ({entry['ENTRYTYPE']}) has no text to "
                "evaluate. It is skipped"
            )
            continue

        logger.debug(f"Found the following sources: {sources}")

        texts.append("\n\n".join(text))

    logger.debug(f"Found {len(texts)} documents")
    return texts


def get_features(
    texts: List[str], feature_count: int, topic_count: int
) -> Tuple[LatentDirichletAllocation, csr_matrix, NDArray[np.string_]]:
    logger.info(f"Calculating count vectorizer for {feature_count} features")
    count_vectorizer = CountVectorizer(stop_words="english", max_features=feature_count)
    term_frequency = count_vectorizer.fit_transform(texts)
    feature_names = count_vectorizer.get_feature_names_out()

    logger.debug(
        f"Found {len(feature_names)},"
        f"\nfeatures: {feature_names},"
        f"\nfrequencies: {term_frequency}"
    )

    logger.info(f"Applying latent dirichlet allocation for {topic_count} components")
    lda = LatentDirichletAllocation(n_components=topic_count)
    lda.fit(term_frequency)

    return lda, term_frequency, feature_names


def flatten(iter: StackedIterable) -> List[Any]:
    values = []
    for value in iter:
        try:
            values += flatten(value)
        except TypeError:
            values.append(value)
    return values


def plot_top_words(
    model: LatentDirichletAllocation, feature_names: NDArray[np.string_]
) -> None:
    topic_count = len(model.components_)
    feature_count = len(feature_names)

    logger.info(
        f"Plotting {topic_count} topics for each of the {feature_count} features"
    )
    fig, axes = plt.subplots(1, topic_count, sharex=True)
    axes = flatten(axes)

    for topic_index, topic in enumerate(model.components_):
        logger.debug(f"Plotting bar graph for topic {topic_index + 1}")

        top_features_indices = topic.argsort()
        logger.debug(f"Index sort of topic is {top_features_indices}")

        top_features = [feature_names[i] for i in top_features_indices]
        logger.debug(f"Top {feature_count} features are {top_features}")

        weights = topic[top_features_indices]
        logger.debug(
            f"Corresponding weights are {feature_count} features are {weights}"
        )

        ax = axes[topic_index]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_index + 1}")

        ax.tick_params(axis="both", which="major")
        for i in "top right left".split():
            ax.spines[i].set_visible(False)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog=NAME,
        description=(
            "Cluster papers into topics according to their titles and/or abstracts"
        ),
    )

    parser.add_argument(
        "--version", "-V", action="version", version=f"{NAME}, version {get_version()}"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="log_level",
        help="Set the loglevel to INFO",
        action="store_const",
        const=INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="log_level",
        help="Set the loglevel to DEBUG",
        action="store_const",
        const=DEBUG,
    )

    parser.add_argument(
        "bibtex_path",
        nargs="?",
        help="The file path of the bibtex file to read",
        type=Path,
    )
    parser.add_argument(
        "-t",
        "--topics",
        type=int,
        help=f"The number of topics, default is {DEFAULT_NUMBER_OF_TOPICS}",
        default=DEFAULT_NUMBER_OF_TOPICS,
        dest="topic_count",
    )
    parser.add_argument(
        "-f",
        "--features",
        type=int,
        help=(
            "The number of features to per topic, default is "
            f"{DEFAULT_NUMBER_OF_FEATURES}"
        ),
        default=DEFAULT_NUMBER_OF_FEATURES,
        dest="feature_count",
    )
    parser.add_argument(
        "--no-title",
        dest="no_title",
        action="store_true",
        help="Use to exclude the title from the feature detection",
        default=False,
    )
    parser.add_argument(
        "--no-abstract",
        dest="no_abstract",
        action="store_true",
        help="Use to exclude the abstract from the feature detection",
        default=False,
    )

    return parser


def main(**kwargs: Unpack[ParserArgs]) -> None:
    parser = get_arg_parser()
    if len(kwargs) > 0:
        args = Namespace(**kwargs)
    else:
        args = parser.parse_args()

    logger.info(
        f"Calculating {args.topic_count} topics with {args.feature_count} features"
    )

    documents = get_documents(args.bibtex_path, not args.no_title, not args.no_abstract)
    model, frequencies, feature_names = get_features(
        documents, args.feature_count, args.topic_count
    )

    plot_top_words(model, feature_names)

    logger.debug("Done, showing plot")
    plt.show()


if __name__ == "__main__":
    main()
