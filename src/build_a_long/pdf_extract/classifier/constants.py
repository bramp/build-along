"""Shared constants for the classifier.

TODO: Consider moving these constants to ClassifierConfig to make them
configurable per analysis run.
"""

# Threshold for classifying a page as a catalog page based on element ID count.
# Pages with more than this many element IDs are considered catalog pages.
CATALOG_ELEMENT_ID_THRESHOLD = 3
