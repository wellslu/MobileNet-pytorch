import unittest

import torch

from src.models import MobileNet


class TestModels(unittest.TestCase):

    def test_resnet(self):
        m = MobileNet()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            y = m(x)

        self.assertListEqual(list(y.size()), [1, 10])


if __name__ == '__main__':
    unittest.main()
