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
                     scared_threshold=10,
                     fear_radius=5,
                     swarm_radius=5):
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

    def get_teleport_action(posStart, posTarget):
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
        if end_position_X < env_bound[0][0]: end_position_X = env_bound[0][0]
        if end_position_X > env_bound[1][0]: end_position_X = env_bound[1][0]
        if end_position_Y < env_bound[0][1]: end_position_Y = env_bound[0][1]
        if end_position_Y > env_bound[1][1]: end_position_Y = env_bound[1][1]
        return (end_position_X, end_position_Y)
    
    def get_escape_action(prey_idx, prey_pos_dict, pred_pos):
        prey_pos = prey_pos_dict[prey_idx]
        # TODO: finish

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
        is_swarm = False
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
        print("pred_pos", pred_pos)
        print("prey_pos", prey_pos_dict)
        print("prey_step_size", prey_step_size)

        # Run experiment, for at most num_steps
        for n in range(1, num_steps):
            print("step ", n)
            herd_direction = get_herd_direction
            # print("is_swarm", is_swarm)
            if is_swarm:
                # calculate mapping from prey to swarm positions
                target_swarm_positions = surround(pred_pos, len(prey_pos_dict), swarm_radius)
                prey_swarm_dict = map_prey_to_swarm(target_swarm_positions, prey_pos_dict)
            
            # print stuff
            # TODO: delete after debugging
            for i, agent in enumerate(agents): 
                if type(agent).__name__ in ["SwarmPreyGrid"]: 
                    print("agent pos: ", prey_pos_dict[i], "isScared: ", agent.isScared)


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
                  if agent.isScared:
                    # try to jump
                    # make other agents around you scared 
                    newly_scared = get_agents_within_fear_radius(prey_pos_dict,i)
                    for scared_agent_idx in newly_scared: 
                        agents[scared_agent_idx].isScared = True

                  else:
                    # update isScared
                    agent.isScared = get_dist(prey_pos_dict[i], pred_pos) < scared_threshold
                  action=herd_direction # TODO: wrong



                else:
                  action = agent(state[i])
                next_state, reward, done, info = env.step(bound_action(state[i],action), i)

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
            print("updated pred_pos", pred_pos)
            print("updated prey_pos_dict", prey_pos_dict)
            # update is_swarm
            is_swarm = update_is_swarm(prey_pos_dict, pred_pos)
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
