import random
from google import genai
from vacuum_local_search import DirtGridEnvironment, PathEntry, MOVES, STOP, MDP_LAYOUT, format_path

# 1. Initialize Q-Learning Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 1.0
living_reward = -0.04

client = genai.Client()

def gemini_hyperparameter_subagent(current_step, current_eps):
    """
    Strategy 4: Sub-agent isolated from main context.
    Returns only the new float value.
    """
    prompt = f"Decay the exploration rate (epsilon). Current step: {current_step}, Current epsilon: {current_eps}. Return ONLY a float decreased by 0.05, minimum 0.1."

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return float(response.text.strip())
    except Exception:
        return current_eps

def run_q_learning_traversal():
    global epsilon
    env = DirtGridEnvironment(MDP_LAYOUT)

    class QAgent:
        def __init__(self, start):
            self.location = start
            self.alive = True
            self.performance = 0
            self.q_table = {}
            
        def get_q(self, state, action):
            return self.q_table.get((state, action), 0.0)

    agent = QAgent(env.start)
    env.add_thing(agent)

    step = 0
    while agent.alive and step < 50:
        percept = env.percept(agent)

        # Sub-agent Delegation
        if step > 0 and step % 5 == 0 and epsilon > 0.1:
            epsilon = gemini_hyperparameter_subagent(step, epsilon)
            print(f"-- Sub-agent updated Epsilon to: {epsilon:.2f} --")

        current_state = agent.location
        
        # 1. Epsilon-Greedy Action Selection
        if random.random() < epsilon:
            action = random.choice(list(MOVES.keys())) # Explore
        else:
            # Exploit: Find the action with the highest Q-value for the current state
            q_values = {a: agent.get_q(current_state, a) for a in MOVES.keys()}
            max_q = max(q_values.values())
            # Tie-breaker for multiple actions with the same max Q-value
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = random.choice(best_actions)
        
        # 2. Execute Action
        env.execute_action(agent, action)
        next_state = agent.location
        
        # 3. Observe Reward (incorporating the living reward penalty)
        raw_reward = env.dirt_at(next_state)
        reward = raw_reward if raw_reward in [10, -1] else living_reward
        
        # 4. Q-Table Update Equation
        # Q(s,a) = Q(s,a) + alpha * [R(s) + gamma * max(Q(s',a')) - Q(s,a)]
        old_q = agent.get_q(current_state, action)
        next_max_q = max([agent.get_q(next_state, a) for a in MOVES.keys()])
        
        new_q = old_q + alpha * (reward + gamma * next_max_q - old_q)
        agent.q_table[(current_state, action)] = new_q

        # Terminal state check
        if raw_reward in [10, -1]:
            agent.alive = False

        step += 1

    print(format_path(env.path_log))

if __name__ == "__main__":
    run_q_learning_traversal()