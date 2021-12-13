from pydantic.dataclasses import dataclass
from pydantic import BaseModel

from typing import Dict, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

# todo add docs for all
# todo remove z.py, validate.py


Survey = Dict[str, Literal[1,2,3,4,5,6,7]]


# todo remove...?
class Request:
    user_id: str


# todo make not strict...
# todo fix
# @dataclass(extra='allow')
@dataclass
class Item:
    """
    Represents a generic item.
    """
    item_id: str
    title: str
    genre: str


@dataclass
class Rating:
    """
    User's rating for an item. `rating` should be a number
    between 1 and 5 (both inclusive).
    """
    item_id: str
    rating: Literal[1,2,3,4,5]


@dataclass
class Preference:
    """
    Represents a predicted or actual preference. `categories`
    is a list of classes that an item belongs to.
    """
    item_id: str
    # categories: List[Literal["CONTROVERSIAL", ""]] # todo more
    categories: Literal["top_N", "diversified_by_latent_feature", "diversified_by_weighted_latent_feature", "diversified_by_emotions", "diversified_by_weighted_emotions"] # todo more

## for data viz
@dataclass
class LatentFeature:
    """
    Represents the latent feature values for visualization
    """
    item_id: str
    feature1: float
    feature2: float
 
@dataclass
class EmotionalSignature:
    """
    Represents the latent feature values for visualization
    """
    item_id: str
    # emotion1: float
    # emotion2: float 
    ############
    emotion1_name: str
    emotion1_val: float
    emotion2_name: str 
    emotion2_val: float 

@dataclass
class Event:
    """
    Represents an interaction with an item.
    """
    item_id: str
    event_type: Literal["hover", "click"]
    duration: int
    enter_time: int
    exit_time: int