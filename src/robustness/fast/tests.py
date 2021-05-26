import unittest

from functions import get_neighbors_ball, get_neighbors_ball_border, get_CI, update_ci
from functions import ID_attack, RD_attack, RD_naive_attack, ICI_attack, RCI_attack

class TestNeigbors(unittest.TestCase):

    def test_get_neighbors_ball_d1(self):
        ## Test graph 001
        nn_set = [
            set([1, 2, 3]),   # 0
            set([0, 4, 5]),   # 1
            set([0, 6, 7]),   # 2
            set([0, 8, 9]),   # 3
            set([1, 9, 10]),  # 4
            set([1, 6, 11]),  # 5
            set([2, 5]),      # 6
            set([2, 8]),      # 7
            set([3, 7, 9]),   # 8
            set([3, 4, 8]),   # 9
            set([4]),         # 10
            set([5])          # 11
        ]
        ball = get_neighbors_ball(nn_set, 0, 1)
        self.assertEqual(set(ball), set([1, 2, 3]))

        ## Wheel graph
        nn_set = [
            set([1, 2, 3, 4]),
            set([0, 2, 4]),
            set([0, 1, 3]),
            set([0, 2, 4]),
            set([0, 1, 3])
        ]
        ball = get_neighbors_ball(nn_set, 0, 1)
        self.assertEqual(set(ball), set([1, 2, 3, 4]))

    def test_get_neighbors_ball_d2(self):
        ## Test graph 001
        nn_set = [
            set([1, 2, 3]),   # 0
            set([0, 4, 5]),   # 1
            set([0, 6, 7]),   # 2
            set([0, 8, 9]),   # 3
            set([1, 9, 10]),  # 4
            set([1, 6, 11]),  # 5
            set([2, 5]),      # 6
            set([2, 8]),      # 7
            set([3, 7, 9]),   # 8
            set([3, 4, 8]),   # 9
            set([4]),         # 10
            set([5])          # 11
        ]
        ball = get_neighbors_ball(nn_set, 0, 2)
        self.assertEqual(set(ball), set([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    def test_get_neighbors_ball_d3(self):
        ## Test graph 001
        nn_set = [
            set([1, 2, 3]),   # 0
            set([0, 4, 5]),   # 1
            set([0, 6, 7]),   # 2
            set([0, 8, 9]),   # 3
            set([1, 9, 10]),  # 4
            set([1, 6, 11]),  # 5
            set([2, 5]),      # 6
            set([2, 8]),      # 7
            set([3, 7, 9]),   # 8
            set([3, 4, 8]),   # 9
            set([4]),         # 10
            set([5])          # 11
        ]
        ball = get_neighbors_ball(nn_set, 0, 3)
        self.assertEqual(set(ball), set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))

    def test_get_neighbors_ball_border_d1(self):
        ## Test graph 001
        nn_set = [
            set([1, 2, 3]),   # 0
            set([0, 4, 5]),   # 1
            set([0, 6, 7]),   # 2
            set([0, 8, 9]),   # 3
            set([1, 9, 10]),  # 4
            set([1, 6, 11]),  # 5
            set([2, 5]),      # 6
            set([2, 8]),      # 7
            set([3, 7, 9]),   # 8
            set([3, 4, 8]),   # 9
            set([4]),         # 10
            set([5])          # 11
        ]
        neighbors = get_neighbors_ball_border(nn_set, 0, 1)
        self.assertEqual(set(neighbors), set([1, 2, 3]))

    def test_get_neighbors_ball_border_d2(self):
        ## Test graph 001
        nn_set = [
            set([1, 2, 3]),   # 0
            set([0, 4, 5]),   # 1
            set([0, 6, 7]),   # 2
            set([0, 8, 9]),   # 3
            set([1, 9, 10]),  # 4
            set([1, 6, 11]),  # 5
            set([2, 5]),      # 6
            set([2, 8]),      # 7
            set([3, 7, 9]),   # 8
            set([3, 4, 8]),   # 9
            set([4]),         # 10
            set([5])          # 11
        ]
        neighbors = get_neighbors_ball_border(nn_set, 0, 2)
        self.assertEqual(set(neighbors), set([4, 5, 6, 7, 8, 9]))

        ## Test graph 003
        nn_set = [
            set([1, 2, 3]), # 0
            set([0, 4, 5]), # 1
            set([0, 6, 7]), # 2
            set([0, 8, 9]), # 3
            set([1, 5, 9]), # 4
            set([1, 4, 6]), # 5
            set([2, 5, 7]), # 6
            set([2, 6, 8]), # 7
            set([3, 7, 9]), # 8
            set([3, 4, 8]), # 9
        ]
        neighbors = get_neighbors_ball_border(nn_set, 0, 2)
        self.assertEqual(set(neighbors), set([4, 5, 6, 7, 8, 9]))

class TestCI(unittest.TestCase):

    def test_CI_l1(self):
        ## Test graph 002
        nn_set = [
            set([1, 2, 3, 4]), # 0
            set([0, 2, 4]),    # 1
            set([0, 1, 3, 6]), # 2
            set([0, 2]),       # 3
            set([0, 1, 5]),    # 4
            set([4]),          # 5
            set([2])           # 6
        ]
        CI_seq = get_CI(nn_set, 1)
        self.assertEqual(CI_seq, [24, 16, 18, 6, 10, 0, 0])

    def test_CI_l2(self):

        ## Test graph 001
        nn_set = [
            set([1, 2, 3]),   # 0
            set([0, 4, 5]),   # 1
            set([0, 6, 7]),   # 2
            set([0, 8, 9]),   # 3
            set([1, 9, 10]),  # 4
            set([1, 6, 11]),  # 5
            set([2, 5]),      # 6
            set([2, 8]),      # 7
            set([3, 7, 9]),   # 8
            set([3, 4, 8]),   # 9
            set([4]),         # 10
            set([5])          # 11
        ]
        CI_seq = get_CI(nn_set, 2)
        self.assertEqual(CI_seq, [20, 14, 16, 14, 16, 12, 5, 7, 12, 10, 0, 0])

        ## Test graph 002
        nn_set = [
            set([1, 2, 3, 4]), # 0
            set([0, 2, 4]),    # 1
            set([0, 1, 3, 6]), # 2
            set([0, 2]),       # 3
            set([0, 1, 5]),    # 4
            set([4]),          # 5
            set([2])           # 6
        ]
        CI_seq = get_CI(nn_set, 2)
        self.assertEqual(CI_seq, [0, 2, 6, 4, 8, 0, 0])

        ## Test graph 003
        nn_set = [
            set([1, 2, 3]), # 0
            set([0, 4, 5]), # 1
            set([0, 6, 7]), # 2
            set([0, 8, 9]), # 3
            set([1, 5, 9]), # 4
            set([1, 4, 6]), # 5
            set([2, 5, 7]), # 6
            set([2, 6, 8]), # 7
            set([3, 7, 9]), # 8
            set([3, 4, 8]), # 9
        ]
        CI_seq = get_CI(nn_set, 2)
        self.assertEqual(CI_seq, [24, 16, 16, 16, 16, 16, 16, 16, 16, 16])

class TestUpdateCI(unittest.TestCase):
    def test_update_ci_1(self):
        ## Test graph 003
        nn_set = [
            set([1, 2, 3]), # 0
            set([0, 4, 5]), # 1
            set([0, 6, 7]), # 2
            set([0, 8, 9]), # 3
            set([1, 5, 9]), # 4
            set([1, 4, 6]), # 5
            set([2, 5, 7]), # 6
            set([2, 6, 8]), # 7
            set([3, 7, 9]), # 8
            set([3, 4, 8]), # 9
        ]
        deg_seq = [len(s) for s in nn_set]
        CI_seq = get_CI(nn_set, l=1)
        v = 0

        w = 1
        true_new_ci_w = 4
        new_ci_w = update_ci(v, w, nn_set, deg_seq, CI_seq, l=1)
        self.assertEqual(new_ci_w, true_new_ci_w)

        w = 4
        true_new_ci_w = 10
        new_ci_w = update_ci(v, w, nn_set, deg_seq, CI_seq, l=1)
        self.assertEqual(new_ci_w, true_new_ci_w)

    def test_update_ci_2(self):
        ## Test graph 003
        nn_set = [
            set([1, 2, 3]), # 0
            set([0, 4, 5]), # 1
            set([0, 6, 7]), # 2
            set([0, 8, 9]), # 3
            set([1, 5, 9]), # 4
            set([1, 4, 6]), # 5
            set([2, 5, 7]), # 6
            set([2, 6, 8]), # 7
            set([3, 7, 9]), # 8
            set([3, 4, 8]), # 9
        ]
        deg_seq = [len(s) for s in nn_set]
        CI_seq = get_CI(nn_set, l=2)
        v = 0
        
        w = 1
        true_new_ci_w = 4
        new_ci_w = update_ci(v, w, nn_set, deg_seq, CI_seq, l=2)
        self.assertEqual(new_ci_w, true_new_ci_w)

        w = 4
        true_new_ci_w = 10
        new_ci_w = update_ci(v, w, nn_set, deg_seq, CI_seq, l=2)
        self.assertEqual(new_ci_w, true_new_ci_w)

class TestAttacks(unittest.TestCase):

    def test_ID_attack(self):
        nn_set = [
            set([3, 4, 5, 6, 7]),
            set([3, 8, 9, 10]),
            set([3, 11]),
            set([0, 1, 2]),
            set([0]),
            set([0]),
            set([0]),
            set([0]),
            set([1]),
            set([1]),
            set([1]),
            set([2]),
        ]
        order = ID_attack(nn_set)
        self.assertEqual(order[:3], [0, 1, 3])

    def test_RD_naive_attack(self):
        nn_set = [
            set([3, 4, 5, 6, 7]),
            set([3, 8, 9, 10]),
            set([3, 11]),
            set([0, 1, 2]),
            set([0]),
            set([0]),
            set([0]),
            set([0]),
            set([1]),
            set([1]),
            set([1]),
            set([2]),
        ]
        order = RD_naive_attack(nn_set)
        self.assertEqual(order[:3], [0, 1, 2])

    def test_RD_attack(self):
        nn_set = [
            set([3, 4, 5, 6, 7]),
            set([3, 8, 9, 10]),
            set([3, 11]),
            set([0, 1, 2]),
            set([0]),
            set([0]),
            set([0]),
            set([0]),
            set([1]),
            set([1]),
            set([1]),
            set([2]),
        ]
        order = RD_attack(nn_set)
        self.assertEqual(order[:3], [0, 1, 2])

    def test_ICI1_attack(self):
        ## Test graph 002
        nn_set = [
            set([1, 2, 3, 4]), # 0
            set([0, 2, 4]),    # 1
            set([0, 1, 3, 6]), # 2
            set([0, 2]),       # 3
            set([0, 1, 5]),    # 4
            set([4]),          # 5
            set([2])           # 6
        ]
        order = ICI_attack(nn_set, l=1)
        self.assertEqual(order[:5], [0, 2, 1, 4, 3])

        nn_set = [
            set([2, 3, 6, 7]),
            set([8, 9]),
            set([0, 4, 5]),
            set([0, 8]),
            set([2]),
            set([2]),
            set([0]),
            set([0]),
            set([1, 3]),
            set([1, 10]),
            set([9])
        ]
        order = ICI_attack(nn_set, l=1)
        self.assertEqual(order[:3], [0, 2, 3])

        nn_set = [
            set([3, 4, 5, 6, 7]),
            set([12, 13, 14, 15]),
            set([18, 19, 20]),
            set([0, 8, 9]),
            set([0, 10]),
            set([0, 11]), # 5
            set([0]),
            set([0]),
            set([3, 27]),
            set([3]),
            set([4]), # 10
            set([5]),
            set([1, 16]),
            set([1, 17]),
            set([1, 28]),
            set([1]), # 15
            set([12]),
            set([13]),
            set([2, 21]),
            set([2, 22]),
            set([2, 23]), # 20
            set([18, 24]),
            set([19, 25]),
            set([20, 26]),
            set([21]),
            set([22]), # 25
            set([23]),
            set([8]),
            set([14])
        ]
        order = ICI_attack(nn_set, l=1)
        self.assertEqual(order[:4], [0, 3, 1, 2])

    def test_ICI2_attack(self):
    
        ## Test graph 001
        nn_set = [
            set([1, 2, 3]),   # 0
            set([0, 4, 5]),   # 1
            set([0, 6, 7]),   # 2
            set([0, 8, 9]),   # 3
            set([1, 9, 10]),  # 4
            set([1, 6, 11]),  # 5
            set([2, 5]),      # 6
            set([2, 8]),      # 7
            set([3, 7, 9]),   # 8
            set([3, 4, 8]),   # 9
            set([4]),         # 10
            set([5])          # 11
        ]
        order = ICI_attack(nn_set, l=2)
        self.assertEqual(order[0], 0)
        self.assertEqual(set(order[1:3]), set([2, 4]))
        self.assertEqual(set(order[3:5]), set([1, 3]))
        self.assertEqual(set(order[5:7]), set([5, 8]))
        self.assertEqual(order[7], 9)

        ## Test graph 002
        nn_set = [
            set([1, 2, 3, 4]), # 0
            set([0, 2, 4]),    # 1
            set([0, 1, 3, 6]), # 2
            set([0, 2]),       # 3
            set([0, 1, 5]),    # 4
            set([4]),          # 5
            set([2])           # 6
        ]

        order = ICI_attack(nn_set, l=2)
        self.assertEqual(order[:4], [4, 2, 3, 1])
    """
    def test_ICI2_attack(self):
        nn_set = [
            set([3, 4, 5, 6, 7]),
            set([12, 13, 14, 15]),
            set([18, 19, 20]),
            set([0, 8, 9]),
            set([0, 10]),
            set([0, 11]), # 5
            set([0]),
            set([0]),
            set([3, 27]),
            set([3]),
            set([4]), # 10
            set([5]),
            set([1, 16]),
            set([1, 17]),
            set([1, 28]),
            set([1]), # 15
            set([12]),
            set([13]),
            set([2, 21]),
            set([2, 22]),
            set([2, 23]), # 20
            set([18, 24]),
            set([19, 25]),
            set([20, 26]),
            set([21]),
            set([22]), # 25
            set([23]),
            set([8]),
            set([14])
        ]
        order = ICI_attack(nn_set, l=2)
        self.assertEqual(order[:4], [0, 3, 2, 1])
    """
    def test_RCI1_attack(self):
        ## Test graph 002
        nn_set = [
            set([1, 2, 3, 4]), # 0
            set([0, 2, 4]),    # 1
            set([0, 1, 3, 6]), # 2
            set([0, 2]),       # 3
            set([0, 1, 5]),    # 4
            set([4]),          # 5
            set([2])           # 6
        ]
        order = RCI_attack(nn_set, l=1)
        self.assertEqual(order[:5], [0, 1, 2, 3, 4])

        nn_set = [
            set([2, 3, 6, 7]),
            set([8, 9]),
            set([0, 4, 5]),
            set([0, 8]),
            set([2]),
            set([2]),
            set([0]),
            set([0]),
            set([1, 3]),
            set([1, 10]),
            set([9])
        ]
        order = RCI_attack(nn_set, l=1)
        self.assertEqual(order[:2], [0, 1])
    
        nn_set = [
            set([3, 4, 5, 6, 7]),
            set([12, 13, 14, 15]),
            set([18, 19, 20]),
            set([0, 8, 9]),
            set([0, 10]),
            set([0, 11]), # 5
            set([0]),
            set([0]),
            set([3, 27]),
            set([3]),
            set([4]), # 10
            set([5]),
            set([1, 16]),
            set([1, 17]),
            set([1, 28]),
            set([1]), # 15
            set([12]),
            set([13]),
            set([2, 21]),
            set([2, 22]),
            set([2, 23]), # 20
            set([18, 24]),
            set([19, 25]),
            set([20, 26]),
            set([21]),
            set([22]), # 25
            set([23]),
            set([8]),
            set([14])
        ]
        order = RCI_attack(nn_set, l=1)
        self.assertEqual(order[:3], [0, 1, 2])
    
    def test_RCI2_attack(self):
    
        ## Test graph 001
        nn_set = [
            set([1, 2, 3]),   # 0
            set([0, 4, 5]),   # 1
            set([0, 6, 7]),   # 2
            set([0, 8, 9]),   # 3
            set([1, 9, 10]),  # 4
            set([1, 6, 11]),  # 5
            set([2, 5]),      # 6
            set([2, 8]),      # 7
            set([3, 7, 9]),   # 8
            set([3, 4, 8]),   # 9
            set([4]),         # 10
            set([5])          # 11
        ]
        order = ICI_attack(nn_set, l=2)
        self.assertEqual(order[:3], [0, 4, 2])

        ## Test graph 002
        nn_set = [
            set([1, 2, 3, 4]), # 0
            set([0, 2, 4]),    # 1
            set([0, 1, 3, 6]), # 2
            set([0, 2]),       # 3
            set([0, 1, 5]),    # 4
            set([4]),          # 5
            set([2])           # 6
        ]

        order = RCI_attack(nn_set, l=2)
        self.assertEqual(order[0], 4)
        self.assertIn(order[1], set([1, 3]))

    """
    def test_RCI2_attack(self):
        nn_set = [
            set([3, 4, 5, 6, 7]),
            set([12, 13, 14, 15]),
            set([18, 19, 20]),
            set([0, 8, 9]),
            set([0, 10]),
            set([0, 11]), # 5
            set([0]),
            set([0]),
            set([3, 27]),
            set([3]),
            set([4]), # 10
            set([5]),
            set([1, 16]),
            set([1, 17]),
            set([1, 28]),
            set([1]), # 15
            set([12]),
            set([13]),
            set([2, 21]),
            set([2, 22]),
            set([2, 23]), # 20
            set([18, 24]),
            set([19, 25]),
            set([20, 26]),
            set([21]),
            set([22]), # 25
            set([23]),
            set([8]),
            set([14])
        ]
        order = RCI_attack(nn_set, l=2)
        self.assertEqual(order[:3], [0, 2, 1])
    """

if __name__ == '__main__':
    unittest.main()
