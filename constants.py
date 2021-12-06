import random

RNG = random.Random()
RNG.seed(42)

DATA_PATH = "data"
INPUT_DIR = "data/input"
EVENTS_DIR = "data/events"
OUTPUT_DIR = "data/output"

EVENTS = ["brexit"]

ALL_COLUMNS = ['archived', 'author', 'author_flair_css_class', 'author_flair_text', 'body', 'controversiality',
               'created_utc', 'distinguished', 'downs', 'edited', 'gilded', 'id', 'link_id', 'name', 'parent_id',
               'removal_reason', 'retrieved_on', 'score', 'score_hidden', 'subreddit', 'subreddit_id', 'ups']

COLUMNS = ["author", "body", "subreddit", "subreddit_id", "created_utc"]

DEM_SUBREDDITS = set([
    "democrats",
    "AOC",
    "Anarchism",
    "AnarchismOnline",
    "AnarchistNews",
    "BadSocialScience",
    "DemocraticSocialism",
    "socialism",
    "Socialism_101",
])

REP_SUBREDDITS = set([
    "republicans",
    "Anarcho_Capitalism",
    "Conservative",
    "ConservativeLounge",
    "ConservativeMeta",
    "DrainTheSwamp",
    "ShitPoliticsSays",
    "Republican",
    "RepublicanValues",
    "progun",
    "antifapassdenied",
    "Capitalism",
    "Libertarian"
])

CEN_SUBREDDITS = ["worldnews", "politics", "news"]

MIN_OCCURENCE_FOR_VOCAB = 50
