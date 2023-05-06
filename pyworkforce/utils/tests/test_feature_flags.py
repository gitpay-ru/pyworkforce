from enum import auto
from pyworkforce.utils.FeatureFlag import FeatureFlag

class MyColor(FeatureFlag):
    Red = auto()
    Green = auto()
    Blue = auto()

    White = Red | Green | Blue


def test_feature_flag_all():

    assert MyColor.Red in MyColor.ALL
    assert MyColor.Green in MyColor.ALL
    assert MyColor.Blue in MyColor.ALL
    assert MyColor.White in MyColor.ALL


def test_feature_flag_none():
    assert MyColor.NONE in MyColor.ALL


def test_deserialize_flags_1_failed():

    try:
        color1 = MyColor.from_str('red')  # should be 'Red' not 'red'

    except AttributeError:
        pass
    else:
        assert False


def test_deserialize_flags_1():

    color1 = MyColor.from_str('Red')

    assert color1 is MyColor.Red

    assert MyColor.Red in color1
    assert MyColor.Blue not in color1
    assert MyColor.Green not in color1

    assert color1 in MyColor.White
    assert MyColor.White not in color1


def test_deserialize_flags_2():

    red_blue = MyColor.from_str('Red Blue')
    assert red_blue is not MyColor.Red
    assert red_blue is not MyColor.Blue

    assert MyColor.Red in red_blue
    assert MyColor.Blue in red_blue
    assert MyColor.Green not in red_blue

    rgb = MyColor.from_str('Blue Red Green')
    assert rgb is not MyColor.Red
    assert rgb is not MyColor.Blue
    assert rgb is not MyColor.Green

    assert MyColor.Red in rgb
    assert MyColor.Blue in rgb
    assert MyColor.Green in rgb

    assert rgb is MyColor.ALL


