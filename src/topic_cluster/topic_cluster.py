from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from bibtexparser import load
from easygui import fileopenbox
from easysettings import EasySettings
from logging import DEBUG, getLogger, StreamHandler
from numpy.typing import NDArray
from os import path, getcwd
from re import compile
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sys import stdout
from tempfile import gettempdir
from typing import Iterable, List, Optional, Protocol, Tuple


NON_ASCII = compile(r"[\W]+")
CONFIG_PATH = path.join(gettempdir(), "topic_cluster.conf")
CONFIG_OPEN_PATH_INDEX = "bibtex_open_path"

logger = getLogger("topic_cluster")


class Model(Protocol):
    components_: Iterable[NDArray[np.float64]]


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


def get_bibtex_path_by_file_open() -> str:
    open_path = get_last_open_file()
    logger.debug("Showing file open dialog")

    open_path = fileopenbox(
        "No bibtex file found",
        "Select bibtex file",
        open_path,
        [[".bibtex", ".bib", "Bibtex files"]],
    )

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

    with open(open_path, mode="r", encoding="utf-8") as bibtex_file:
        db = load(bibtex_file)

    texts = []
    keys = (["title"] if title else []) + (["abstract"] if abstract else [])
    sources = defaultdict(int)

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
                f"The entry {entry['ID']} ({entry['ENTRYTYPE']}) has no text to evaluate. It is skipped"
            )
            continue

        logger.debug(f"Found the following sources: {sources}")

        texts.append("\n\n".join(text))

    logger.debug(f"Found {len(texts)} documents")
    return texts


def get_features(
    texts: List[str], feature_count: int, topic_count: int
) -> Tuple[Model, NDArray, NDArray[np.string_]]:
    logger.info(f"Calculating count vectorizer for {feature_count} features")
    count_vectorizer = CountVectorizer(stop_words="english", max_features=feature_count)
    term_frequency: NDArray = count_vectorizer.fit_transform(texts)
    feature_names = count_vectorizer.get_feature_names_out()

    logger.debug(
        f"Found {len(feature_names)},\nfeatures: {feature_names},\nfrequencies: {term_frequency}"
    )

    logger.info(f"Applying latent dirichlet allocation for {topic_count} components")
    lda = LatentDirichletAllocation(n_components=topic_count)
    lda.fit(term_frequency)

    return lda, term_frequency, feature_names


def plot_top_words(
    model: Model, feature_names: List[str]
) -> None:
    topic_count = len(model.components_)
    feature_count = len(feature_names)

    logger.info(
        f"Plotting {topic_count} topics for each of the {feature_count} features"
    )
    fig, axes = plt.subplots(1, topic_count, sharex=True)
    axes = axes.flatten()

    for topic_index, topic in enumerate(model.components_):
        logger.debug(f"Plotting bar graph for topic {topic_index + 1}")

        top_features_indices = topic.argsort()
        logger.debug(f"Index sort of topic is {top_features_indices}")

        top_features = [feature_names[i] for i in top_features_indices]
        logger.debug(f"Top {feature_count} features are {top_features}")

        weights = topic[top_features_indices]
        logger.debug(f"Corresponding weights are {feature_count} features are {weights}")

        ax = axes[topic_index]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_index + 1}")

        ax.tick_params(axis="both", which="major")
        for i in "top right left".split():
            ax.spines[i].set_visible(False)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)


def main(
    open_path: Optional[str] = None,
    use_title: bool = True,
    use_abstract: bool = True,
    feature_count: int = 10,
    topic_count: int = 3,
) -> None:
    logger.info(f"Calculating {topic_count} topics with {feature_count} features")

    documents = get_documents(open_path, use_title, use_abstract)
    model, frequencies, feature_names = get_features(
        documents, feature_count, topic_count
    )

    plot_top_words(model, feature_names)

    logger.debug("Done, showing plot")
    plt.show()


if __name__ == "__main__":
    handler = StreamHandler(stdout)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    main(feature_count=7, topic_count=4)
