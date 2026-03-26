def integrator_step_ffi(state, dt):
    state["position"] = state["position"] + state["velocity"] * dt
    return state

