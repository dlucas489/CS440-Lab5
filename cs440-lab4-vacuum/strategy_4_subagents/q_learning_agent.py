import random
import re
from google import genai
from aima.agents import Agent
from vacuum_local_search import DirtGridEnvironment, PathEntry, MOVES, STOP, MDP_LAYOUT, format_path

# 1. Initialize Q-Learning Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 1.0
living_reward = -0.04

client = genai.Client()

def gemini_hyperparameter_subagent(current_step, current_eps):
    prompt = f"Calculate {current_eps} - 0.05. Return ONLY the raw number. Do not include markdown or words."
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        match = re.search(r"0\.\d+", response.text)
        return float(match.group()) if match else current_eps
    except Exception:
        return round(max(0.1, current_eps - 0.05), 2)

def run_q_learning_traversal():
    global epsilon
    env = DirtGridEnvironment(MDP_LAYOUT)

    class QAgent(Agent):
        def __init__(self, start):
            super().__init__(program=lambda percept: None)
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

        if step > 0 and step % 5 == 0 and epsilon > 0.1:
            epsilon = gemini_hyperparameter_subagent(step, epsilon)
            print(f"-- Sub-agent updated Epsilon to: {epsilon:.2f} --")

        current_state = agent.location

        if random.random() < epsilon:
            action = random.choice(list(MOVES.keys())) # Explore
        else:
            q_values = {a: agent.get_q(current_state, a) for a in MOVES.keys()}
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = random.choice(best_actions)

        env.execute_action(agent, action)
        next_state = agent.location

        raw_reward = env.dirt_at(next_state)
        reward = raw_reward if raw_reward in [10, -1] else living_reward

        old_q = agent.get_q(current_state, action)
        next_max_q = max([agent.get_q(next_state, a) for a in MOVES.keys()])
        
        new_q = old_q + alpha * (reward + gamma * next_max_q - old_q)
        agent.q_table[(current_state, action)] = new_q

        if raw_reward in [10, -1]:
            agent.alive = False

        step += 1

    print(format_path(env.path_log))

if __name__ == "__main__":
    run_q_learning_traversal()