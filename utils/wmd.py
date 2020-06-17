from itertools import product
from collections import defaultdict
from scipy.spatial.distance import euclidean
import numpy as np
import pulp


class WMD:
    def __tokens_to_fracdict__(self, tokens):
        cntdict = defaultdict(lambda: 0)
        for token in tokens:
            cntdict[token] += 1
        totalcnt = sum(cntdict.values())
        return {token: float(cnt) / totalcnt for token, cnt in cntdict.items()}

    # use PuLP
    def word_mover_distance_probspec(self, first_sent_tokens, second_sent_tokens, wvmodel, lpFile=None):

        all_tokens = list(set(first_sent_tokens + second_sent_tokens))
        wordvecs = {token: wvmodel[token] for token in all_tokens}

        first_sent_buckets = self.__tokens_to_fracdict__(first_sent_tokens)
        second_sent_buckets = self.__tokens_to_fracdict__(second_sent_tokens)

        T = pulp.LpVariable.dicts('T_matrix', list(product(all_tokens, all_tokens)), lowBound=0)

        prob = pulp.LpProblem('WMD', sense=pulp.LpMinimize)
        prob += pulp.lpSum([T[token1, token2] * euclidean(wordvecs[token1], wordvecs[token2])
                            for token1, token2 in product(all_tokens, all_tokens)])
        for token2 in second_sent_buckets:
            prob += pulp.lpSum([T[token1, token2] for token1 in first_sent_buckets]) == second_sent_buckets[token2]
        for token1 in first_sent_buckets:
            prob += pulp.lpSum([T[token1, token2] for token2 in second_sent_buckets]) == first_sent_buckets[token1]

        if lpFile != None:
            prob.writeLP(lpFile)

        prob.solve()

        return prob

    def word_mover_distance(self, first_sent_tokens, second_sent_tokens, wvmodel, lpFile=None):
        prob = self.word_mover_distance_probspec(first_sent_tokens, second_sent_tokens, wvmodel, lpFile=lpFile)
        return pulp.value(prob.objective)

    def trans_matrix(self, first_sent_tokens, second_sent_tokens, wvmodel, lpFile=None):
        prob = self.word_mover_distance_probspec(first_sent_tokens, second_sent_tokens, wvmodel, lpFile=lpFile)

        first_sent_indexes = {t: i for i, t in enumerate(first_sent_tokens)}
        second_sent_indexes = {t: i for i, t in enumerate(second_sent_tokens)}

        result = np.zeros((len(first_sent_indexes), len(second_sent_indexes)), dtype=np.float)

        for v in prob.variables():
            if v.varValue != 0:
                token_temp = v.name.split('\'')
                token1 = token_temp[1]
                token2 = token_temp[3]

                if token1 not in first_sent_indexes or token2 not in second_sent_indexes:
                    raise ValueError

                result[first_sent_indexes[token1], second_sent_indexes[token2]] = v.varValue

        return result
