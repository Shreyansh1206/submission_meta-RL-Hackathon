import os
import sys
import math
import contextlib
from fastapi import FastAPI
import uvicorn
import threading

app = FastAPI()

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w', encoding='utf-8') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

from sumo_env import SumoTrafficEnv
from agent import DQNAgent

def scale_reward_for_printing(reward):
    try:
        return max(0.0, min(1.0, math.exp(float(reward) / 100.0)))
    except:
        return 0.0

@app.get("/")
def root():
    return {"status": "running", "message": "RL Traffic Lights Controller"}

@app.post("/reset")
def reset():
    return {"status": "ok"}

@app.post("/step")
def step(data: dict = {}):
    return {"status": "ok"}

@app.post("/close")
def close():
    return {"status": "ok"}

@app.post("/run")
def run():
    thread = threading.Thread(target=run_inference)
    thread.start()
    return {"status": "started"}

def run_inference():
    task_name = "traffic-light-optimization"
    env_name = "SumoTrafficEnv"
    model_name = "DQNAgent"

    print(f"[START] task={task_name} env={env_name} model={model_name}")

    rewards_history = []
    steps = 0
    success = False
    last_action_error = None

    try:
        with suppress_stdout():
            env = SumoTrafficEnv(gui=False)
            agent = DQNAgent(obs_dim=30, n_actions=4)
            if os.path.exists("best_traffic_agent.pth"):
                agent.load("best_traffic_agent.pth")
            agent.epsilon = 0.0
            state, _ = env.reset()

        done = False
        while not done:
            with suppress_stdout():
                action = agent.act(state)
            steps += 1
            try:
                with suppress_stdout():
                    state, internal_reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                last_action_error = None
            except Exception as e:
                last_action_error = str(e).replace('\n', ' ').strip()
                internal_reward = 0.0
                done = True

            rewards_history.append(internal_reward)
            print_r = scale_reward_for_printing(internal_reward)
            done_str = "true" if done else "false"
            err_str = f'"{last_action_error}"' if last_action_error else "null"
            print(f"[STEP] step={steps} action={action} reward={print_r:.2f} done={done_str} error={err_str}")

        success = not last_action_error

    except Exception as e:
        success = False
        if not last_action_error:
            last_action_error = str(e).replace('\n', ' ').strip()
    finally:
        with suppress_stdout():
            try:
                env.close()
            except:
                pass
        success_str = "true" if success else "false"
        formatted_rewards = ",".join([f"{scale_reward_for_printing(r):.2f}" for r in rewards_history])
        print(f"[END] success={success_str} steps={steps} rewards={formatted_rewards}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)