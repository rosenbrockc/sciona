class RollingAveragerAtom:
    def inject(self, state, value):
        state.buffer.append(value)
        state.count += 1
        return state

    def run(self, state):
        if not state.buffer:
            return 0.0
        return sum(state.buffer) / len(state.buffer)

