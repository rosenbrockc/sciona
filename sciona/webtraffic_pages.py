"""Web Traffic page metadata and agent grouping; MIT source adaptation.

Derived from Arturus/kaggle-web-traffic; see docs/licenses/WebTraffic-MIT.txt.
Page strings are explicit runtime inputs. No observed page names are bundled.
"""
import re
from typing import Collection, Dict
import numpy as np
import pandas as pd

term_pat = re.compile('(.+?):(.+)')

pat = re.compile('(.+)_([a-z][a-z]\\.)?((?:wikipedia\\.org)|(?:commons\\.wikimedia\\.org)|(?:www\\.mediawiki\\.org))_([a-z_-]+?)$')

def extract_page_fields(source) -> pd.DataFrame:
    """
    Extracts features from url. Features: agent, site, country, term, marker
    :param source: urls
    :return: DataFrame, one column per feature
    """
    if isinstance(source, pd.Series):
        source = source.values
    agents = np.full_like(source, np.nan)
    sites = np.full_like(source, np.nan)
    countries = np.full_like(source, np.nan)
    terms = np.full_like(source, np.nan)
    markers = np.full_like(source, np.nan)
    for i in range(len(source)):
        l = source[i]
        match = pat.fullmatch(l)
        assert match, 'Non-matched string %s' % l
        term = match.group(1)
        country = match.group(2)
        if country:
            countries[i] = country[:-1]
        site = match.group(3)
        sites[i] = site
        agents[i] = match.group(4)
        if site != 'wikipedia.org':
            term_match = term_pat.match(term)
            if term_match:
                markers[i] = term_match.group(1)
                term = term_match.group(2)
        terms[i] = term
    return pd.DataFrame({'agent': agents, 'site': sites, 'country': countries, 'term': terms, 'marker': markers, 'page': source})

def make_page_features(pages: np.ndarray) -> pd.DataFrame:
    """
    Calculates page features (site, country, agent, etc) from urls
    :param pages: Source urls
    :return: DataFrame with features as columns and urls as index
    """
    tagged = extract_page_fields(pages).set_index('page')
    features: pd.DataFrame = tagged.drop(['term', 'marker'], axis=1)
    return features

def uniq_page_map(pages: Collection):
    """
    Finds agent types (spider, desktop, mobile, all) for each unique url, i.e. groups pages by agents
    :param pages: all urls (must be presorted)
    :return: array[num_unique_urls, 4], where each column corresponds to agent type and each row corresponds to unique url.
     Value is an index of page in source pages array. If agent is missing, value is -1
    """
    import re
    result = np.full([len(pages), 4], -1, dtype=np.int32)
    pat = re.compile('(.+(?:(?:wikipedia\\.org)|(?:commons\\.wikimedia\\.org)|(?:www\\.mediawiki\\.org)))_([a-z_-]+?)')
    prev_page = None
    num_page = -1
    agents = {'all-access_spider': 0, 'desktop_all-agents': 1, 'mobile-web_all-agents': 2, 'all-access_all-agents': 3}
    for i, entity in enumerate(pages):
        match = pat.fullmatch(entity)
        assert match
        page = match.group(1)
        agent = match.group(2)
        if page != prev_page:
            prev_page = page
            num_page += 1
        result[num_page, agents[agent]] = i
    return result[:num_page + 1]

def encode_page_features(df) -> Dict[str, pd.DataFrame]:
    """
    Applies one-hot encoding to page features and normalises result
    :param df: page features DataFrame (one column per feature)
    :return: dictionary feature_name:encoded_values. Encoded values is [n_pages,n_values] array
    """

    def encode(column) -> pd.DataFrame:
        one_hot = pd.get_dummies(df[column], drop_first=False)
        return (one_hot - one_hot.mean()) / one_hot.std()
    return {str(column): encode(column) for column in df}
