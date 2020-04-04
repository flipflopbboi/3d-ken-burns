from collections import OrderedDict
from typing import Callable, Iterable, List, MutableMapping, Type, TypeVar

S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def list_to_dict_list(
    list_: Iterable[T],
    key_func: Callable[[T], S],
    item_func: Callable[[T], U] = lambda obj: obj,
    dict_type: Type[MutableMapping] = OrderedDict,
) -> MutableMapping[S, List[U]]:
    """
    Returns a dictionary of lists based on the given list

    Parameters
    ----------
    list_
        The list to convert
    key_func
        A function that returns the key to be used in the dictionary.
    item_func
        A function that returns the value to be used in the dictionary.
        Defaults to a function that simply returns the item as it is in the list
    dict_type
        The type of mapping to return. This allows one to optionally an unordered dict or custom mapping type.
    """
    result = dict_type()
    for list_item in list_:
        key = key_func(list_item)
        dict_item = item_func(list_item)
        if key in result:
            result[key].append(dict_item)
        else:
            result[key] = [dict_item]
    return result
