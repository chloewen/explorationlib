import os
import numpy as np
import math
import random 
# import cloudpickle

from copy import deepcopy
from collections import defaultdict

from explorationlib.util import save
from tqdm.autonotebook import tqdm
from explorationlib import local_gym
from explorationlib import agent as agent_gym



def experiment(name,
               agent,
               env,
               num_steps=1,
               num_experiments=1,
               seed=None,
               split_state=False,
               dump=True,
               env_kwargs=None,
               agent_kwargs=None):
    """Run an experiment. 
    
    Note: by default the experiment log gets saved to 'name' and this
    function returns None: To return the exp_data, set dump=False.
    """

    # Parse env
    if isinstance(env, str):
        Env = getattr(local_gym, env)
        if env_kwargs is not None:
            env = Env(**env_kwargs)
        else:
            env = Env()

    # Parse agent
    if isinstance(agent, str):
        Agent = getattr(agent_gym, agent)
        if agent_kwargs is not None:
            agent = Agent(**agent_kwargs)
        else:
            agent = Agent()

    # Pretty name
    base = os.path.basename(name)
    base = os.path.splitext(base)[0]

    # Seed
    agent.seed(seed)
    env.seed(seed)

    # Add one log for each exp
    # to the results list
    results = []

    # !
    for k in tqdm(range(num_experiments), desc=base):
        # Create an exp log
        log = defaultdict(list)

        # Reset world
        agent.reset()
        env.reset()
        state, reward, done, info = env.last()

        # Run experiment, for at most num_steps
        for n in range(1, num_steps):
            # Step
            action = agent(state)
            env.step(action)
            next_state, reward, done, info = env.last()

            # Learn? Might do nothing.
            agent.update(state, action, reward, next_state, info)

            # Shift
            state = deepcopy(next_state)

            # Log step env
            log["exp_step"].append(deepcopy(n))
            log["num_experiment"].append(deepcopy(k))
            if split_state:
                pos, obs = state
                log["exp_state"].append(deepcopy(pos))
                log["exp_obs"].append(deepcopy(obs))
            else:
                log["exp_state"].append(deepcopy(state))
            log["exp_action"].append(deepcopy(action))
            log["exp_reward"].append(deepcopy(reward))
            log["exp_info"].append(deepcopy(info))

            if done:
                break

        # Metadata
        log["exp_agent"] = deepcopy(agent)
        log["exp_name"] = base
        log["num_experiments"] = num_experiments
        log["exp_num_steps"] = num_steps
        log["exp_env"] = env

        # Log full agent history
        # TODO - someday update all code to save comp and reg exps the same
        # way, like this
        # log["agent_history"] = []
        # log["agent_history"].append(agent.history)

        # Old fmt. Replace w/ above
        for k in agent.history.keys():
            log[k].extend(deepcopy(agent.history[k]))

        # Save the log to the results
        results.append(log)

    if dump:
        if not name.endswith(".pkl"):
            name += ".pkl"
        save(results, filename=name)
    else:
        return results

def multi_experiment(name,
                     agents,
                     env,
                     env_bound=[(-1*math.inf, -1*math.inf), (math.inf, math.inf)],
                     num_steps=1,
                     num_experiments=5, # maybe
                     seed=None,
                     split_state=False,
                     dump=True,
                     env_kwargs=None,
                     fear_radius=5,
                     prey_radius=5,
                     pred_radius=10):
    """Run a multi-agent experiment. Targets can also be agents. 
    
    Note: by default the experiment log gets saved to 'name' and this
    function returns None: To return the exp_data, set dump=False.
    """

    def get_dist(posA, posB):
        return math.sqrt((posA[0]-posB[0])**2 + (posA[1]-posB[1])**2)

    # Returns a list of tuples of len preycount containing coordinates equidistant to predCoords by a distance of radius
    def surround(pred_pos, prey_count, radius):
        result = []

        # Angle between each new point
        angle_step = 2 * math.pi / prey_count

        # Calculate coordinates for each point and append to result
        for i in range(prey_count):
            angle = i * angle_step
            x_coord = pred_pos[0] + radius * math.floor(100 * (math.cos(angle))) / 100
            y_coord = pred_pos[1] + radius * math.floor(100 * (math.sin(angle))) / 100
            result.append((x_coord, y_coord))

        return result
    
    # returns a dict mapping prey index to closest swarm position 
    def map_prey_to_swarm(swarm_pos_list, prey_pos_dict):
        prey_swarm_dict = dict()
        for prey_index in prey_pos_dict: 
            prey_pos = prey_pos_dict[prey_index]
            dist_to_swarm = [get_dist(prey_pos, swarm_pos) for swarm_pos in swarm_pos_list]
            closest_swarm_pos_index = dist_to_swarm.index(min(dist_to_swarm))
            closest_swarm_pos = swarm_pos_list[closest_swarm_pos_index]
            prey_swarm_dict[prey_index] = closest_swarm_pos
            swarm_pos_list.remove(closest_swarm_pos)
        return prey_swarm_dict 

    # returns True is at least 1 prey is within swarm_threshold of predator
    def update_is_swarm(prey_pos_dict, pred_pos):
        for prey_idx in prey_pos_dict:
            prey_pos = prey_pos_dict[prey_idx]
            if get_dist(prey_pos, pred_pos) < scared_threshold:
                print("fraud prey: ", prey_idx, "pos:" , prey_pos)
                print("pred_pos", pred_pos)
                return True
        return False

    def get_action(posStart, posTarget):
        return [posTarget[0]-posStart[0], posTarget[1]-posStart[1]]

    # returns list of indices of prey within fear radius of prey pred_idx
    def get_agents_within_fear_radius(prey_pos_dict, scared_prey_idx):
        res = []
        for prey_idx in prey_pos_dict:
            if prey_idx != scared_prey_idx and get_dist(prey_pos_dict[prey_idx], prey_pos_dict[scared_prey_idx]) < fear_radius:
                res += [prey_idx]
        return res

    def get_herd_direction(prey_step_size):
        poss_actions = [[-1 * prey_step_size,0], [prey_step_size,0], [0,-1 * prey_step_size], [0,prey_step_size]]
        return random.choice(poss_actions)
    
    def bound_action(pos, action):
        end_position_X = pos[0] + action[0]
        end_position_Y = pos[1] + action[1]
        # TODO: this is bad 
        if end_position_X < env_bound[0][0]: end_position_X = env_bound[0][0]
        if end_position_X > env_bound[1][0]: end_position_X = env_bound[1][0]
        if end_position_Y < env_bound[0][1]: end_position_Y = env_bound[0][1]
        if end_position_Y > env_bound[1][1]: end_position_Y = env_bound[1][1]
        return (end_position_X-pos[0], end_position_Y-pos[1])
    
    def in_bounds(pos):
        return (pos[0] >= env_bound[0][0] 
                and pos[0] <= env_bound[1][0] 
                and pos[1] >= env_bound[0][1] 
                and pos[1] <= env_bound[1][1])
    
    def is_valid(prey_pos_dict, pred_pos):
        for prey_a_idx in prey_pos_dict:
            # check if all prey are within bounds
            if not in_bounds(pred_pos): return False
            # check if all prey are far enough from the predator
            if get_dist(prey_pos_dict[prey_a_idx], pred_pos) < pred_radius: return False
            for prey_b_idx in prey_pos_dict: 
            # check if all prey are far enough from each other 
                if prey_a_idx != prey_b_idx:
                    if get_dist(prey_pos_dict[prey_a_idx], prey_pos_dict[prey_b_idx]) < prey_radius: return False
        return True
    
    def get_valid_action(prey_idx, target_pos, prey_pos_dict, pred_pos, step_size):
        try_incrs = [i * math.pi/36 for i in range(36)]
        prey_pos = prey_pos_dict[prey_idx]
        escape_angle = math.atan((target_pos[1]-prey_pos[1]) / (target_pos[0]-prey_pos[0])) + math.pi
        for try_incr in try_incrs:
            # try moving 5 degrees one way
            try_angle = escape_angle + try_incr
            x_new = prey_pos[0] + step_size * math.floor(100 * (math.cos(try_angle))) / 100
            y_new = prey_pos[1] + step_size * math.floor(100 * (math.sin(try_angle))) / 100
            new_pos = [x_new, y_new]
            prey_pos_dict[prey_idx] = new_pos
            
            if is_valid(prey_pos_dict, pred_pos): 
                return get_action(prey_pos, new_pos)
            # try moving 5 degrees the opposite way
            try_angle = escape_angle - try_incr
            x_new = prey_pos[0] + step_size * math.floor(100 * (math.cos(try_angle))) / 100
            y_new = prey_pos[1] + step_size * math.floor(100 * (math.sin(try_angle))) / 100
            new_pos = [x_new, y_new]
            prey_pos_dict[prey_idx] = new_pos
            if is_valid(prey_pos_dict, pred_pos): 
                return get_action(prey_pos, new_pos)
        # if no valid moves, don't move
        print("no valid moves")
        return [0,0]

    def get_pos_from_action(pos, action):
        return [pos[0]+action[0], pos[1]+action[1]]
   

    # Parse env
    if isinstance(env, str):
        Env = getattr(local_gym, env)
        if env_kwargs is not None:
            env = Env(**env_kwargs)
        else:
            env = Env()

    # Pretty name
    base = os.path.basename(name)
    base = os.path.splitext(base)[0]

    # Seed
    if seed is not None:
        [agent.seed(seed + i) for i, agent in enumerate(agents)]
        env.seed(seed)

    # Add one log for each exp
    # to the results list
    results = []

    # !
    for k in tqdm(range(num_experiments), desc=base):
        # is_swarm = False
        pred_pos = None
        prey_pos_dict = dict()
        prey_step_size = 1

        # Create an exp log
        log = defaultdict(list)

        # Reset agents...
        [agent.reset() for agent in agents]

        # and the world
        env.reset()
        state, reward, done, info = env.last()
        print("state", state)

        # get initial positions of all agents, step size
        for i,agent in enumerate(agents):
            if type(agent).__name__ in ["GreedyPredatorGrid"]:
                pred_pos = state[i]
            elif type(agent).__name__ in ["SwarmPreyGrid"]: 
                prey_pos_dict[i] = state[i]
                prey_step_size = agent.step_size

        # Run experiment, for at most num_steps
        for n in range(1, num_steps):
            print("step ", n)
            herd_direction = get_herd_direction(prey_step_size)
            print("env.dead", env.dead)
            print("pred_pos", pred_pos)
            print("swarm prey")

            for i, agent in enumerate(agents):
                # The dead don't step
                if i in env.dead:
                    continue

                # Step the agent
                if type(agent).__name__ in ["GreedyPredatorGrid"]:
                  # state is current agent state & other agent states
                  state_ = [state[i], [
                      x for i_, x in enumerate(state) if i_ != i]]
                  action = agent(state_)
                elif type(agent).__name__ in ["SwarmPreyGrid"]: 
                  print("idx", i, "| prev pos", prey_pos_dict[i], "| isScared", agent.isScared, end=" ")
                  if agent.isScared:
                    # make other agents around you scared 
                    newly_scared = get_agents_within_fear_radius(prey_pos_dict,i)
                    for scared_agent_idx in newly_scared: 
                        agents[scared_agent_idx].isScared = True
                    # try to move away from the predator fast 
                    action = get_valid_action(i, prey_pos_dict, pred_pos, prey_step_size * 2)
                  else:
                    # try to move with the herd
                    herd_final_pos = get_pos_from_action(prey_pos_dict[i], herd_direction)
                    action = get_valid_action(i, prey_pos_dict, herd_final_pos, prey_step_size)
                    # TODO: state where prey aren't together
                  print("| action", action, end=" ")

                  # update isScared
                  next_pos = [prey_pos_dict[i][0] + action[0], prey_pos_dict[i][1] + action[1]]
                  agent.isScared = get_dist(next_pos, pred_pos) <= fear_radius
                else:
                  action = agent(state[i])
                next_state, reward, done, info = env.step(action, i)
                print("| next pos", next_state[i])

                # update pred_pos, prey_pos_dict
                next_pos = next_state[i]
                if type(agent).__name__ in ["GreedyPredatorGrid"]:
                    pred_pos = next_pos
                elif type(agent).__name__ in ["SwarmPreyGrid"]:
                    # update prey_pos
                    prey_pos_dict[i] = next_pos

                # Learn? Might do nothing.
                agent.update(state, action, reward, next_state, info)

                # Shift
                state = deepcopy(next_state)

                # Log step env
                log["num_experiment"].append(deepcopy(k))
                log["exp_step"].append(deepcopy(n))
                log["exp_agent"].append(deepcopy(i))
                log["exp_action"].append(deepcopy(action))
                log["exp_reward"].append(deepcopy(reward))
                log["exp_info"].append(deepcopy(info))

                # Lod dead, if env has this
                try:
                    log["exp_env_dead"].append(deepcopy(env.dead))
                except AttributeError:
                    pass

                # Are there senses obs?
                if split_state:
                    pos, obs = state
                    log["exp_state"].append(deepcopy(pos))
                    log["exp_obs"].append(deepcopy(obs))
                else:
                    log["exp_state"].append(deepcopy(state))

                # ?
                if done:
                    break
        # Save agent and env
        log["exp_agent"] = deepcopy(agent)

        # Log agents history
        log["agent_history"] = []
        for agent in agents:
            log["agent_history"].append(agent.history)

        # Save the log to the results
        results.append(log)

    # Metadata
    log["exp_name"] = base
    log["num_experiments"] = num_experiments
    log["exp_num_steps"] = num_steps
    log["env"] = env

    if dump:
        if not name.endswith(".pkl"):
            name += ".pkl"
        save(results, filename=name)
    else:
        return results


if __name__ == "__main__":
    import fire
    fire.Fire({"experiment": experiment})
