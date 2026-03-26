class PosteriorAccumulatorAtom:
    def run(self, alpha, beta, successes, failures):
        return {
            "alpha": alpha + successes,
            "beta": beta + failures,
        }

