from unittest import TestCase, main
from roma.imp import shifts_duration


class ShiftsDurationTest(TestCase):
    def test_one(self):
        self.assertEqual(shifts_duration('32400'), 32400)


if __name__ == '__main__':
    main()
