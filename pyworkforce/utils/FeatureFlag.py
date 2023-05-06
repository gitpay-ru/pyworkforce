from enum import Flag
from functools import reduce
from operator import or_ as _or_


class classproperty:
    def __init__(self, func):
        self._func = func
    def __get__(self, obj, owner):
        return self._func(owner)


class FeatureFlag(Flag):
    @classproperty
    def NONE(cls):
        none_mbr = cls(0)

        cls._member_map_['NONE'] = none_mbr
        return none_mbr

    @classproperty
    def ALL(cls):
        cls_name = cls.__name__
        if not len(cls):
            raise AttributeError('empty %s does not have an ALL value' % cls_name)

        all_mbr = cls(reduce(_or_, cls))
        # for member in cls.__members__.values():
        #     all_mbr |= member

        cls._member_map_['ALL'] = all_mbr
        return all_mbr

    @classmethod
    def from_str(cls, enum_str: str):
        retval = cls.NONE
        for name in enum_str.split():
            retval |= cls.__getattr__(name)

        return retval

