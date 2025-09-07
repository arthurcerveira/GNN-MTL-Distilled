import warnings
warnings.simplefilter("ignore")

try:
    import flaml
except AttributeError:
    # raises AttributeError: readonly attribute in functools.
    # For some reason, importing flaml again fixes the issue.
    import flaml

from . import download
from . import predict
from . import preprocess
from . import train
from . import validate