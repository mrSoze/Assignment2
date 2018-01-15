import numpy as np


class MCMCSampler(object):

    def get_proposal(self):
        tmp = np.random.randint(1, 4)
        proposal_state=None
        if tmp == 1:
            proposal_state = self.first_proposal()
        elif tmp ==2:
            proposal_state = self.second_proposal()
        elif tmp == 3:
            proposal_state = self.third_proposal()
        else:
            raise("random out of bound")
        return proposal_state, tmp

    def first_proposal(self):
