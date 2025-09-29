def reward_function(func):
    """Decorator to mark functions as reward functions."""
    func._is_reward_function = True
    return func
