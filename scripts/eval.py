# from utils.env import QuestJerichoEnv
# from quest_interface.heuristic_agents import RandomAgent, RepeatAgent, HumanAgent, Seq2SeqAgent
from handmethat.envs.jericho_env import HMTJerichoEnv
from handmethat.envs.env import HMTEnv
from handmethat.model.random.random_agent import RandomAgent

import json
import numpy as np


def evaluate(agent, fully, num=None, level=None):
    import ipdb; ipdb.set_trace()
    eval_results = list()
    step_limit = 40
    working_dir = './data/'
    dataset_name = 'HandMeThat_with_expert_demonstration'
    data_info = working_dir + 'HandMeThat_data_info.json'
    with open(data_info, 'r') as f:
        json_str = json.load(f)
    validate = json_str[2]['test']
    if level:
        validate = [file for file in validate if level in file]
    validate = np.random.permutation(validate)
    # import ipdb; ipdb.set_trace()
    if num:
        validate = validate[:num]
    for filename in validate:
        # import ipdb; ipdb.set_trace()
        path = working_dir + dataset_name + '/' + filename
        eval_env = HMTJerichoEnv(path, None, fully, step_limit=step_limit, get_valid=True)
        print('Current task file:', eval_env.json_path)

        obs, info = eval_env.reset()

        # simplify for training
        idx_start = obs.find('The human agent has')
        idx_end = obs.find('Now you are')
        obs = obs[:idx_start] + obs[idx_end:]
        obs = obs.replace('#', ' ')
        obs = obs.replace(',', '')

        done = False
        reward = 0
        agent.reset(eval_env.env)
        actions = list()
        for _ in range(step_limit):
            action = agent.act(obs, reward, done, info)
            actions.append(action)
            ob, reward, done, info = eval_env.step(action)
            print('Action:', action)
            print('Effect:', ob)
            ob = ob.replace('#', ' ')  # simplify: remove '#'
            ob = ob.replace(',', '')
            obs += ' [SEP] ' + action + ' [SEP] '
            obs += ob
            # import ipdb; ipdb.set_trace()
            if not fully:
                obs += eval_env.env.get_look(fully=False).replace('#', ' ')
            if done:
                break
        if reward > 0:
            print('Succeed!')
        else:
            print('Fail!')
        eval_results.append((info['moves'], info['score'], info['question_cost']))
        print('moves: ', info['moves'], '; score: ', info['score'])
    print(eval_results)
    return eval_results


def evaluate_for_seq2seq(agent, fully, num=None, level=None):
    np.random.seed(2)
    eval_results = list()
    step_limit = 40
    working_dir = './data/'
    dataset_name = 'HandMeThat_with_expert_demonstration'
    data_info = working_dir + 'HandMeThat_data_info.json'
    with open(data_info, 'r') as f:
        json_str = json.load(f)
    validate = json_str[2]['test']
    if level:
        validate = [file for file in validate if level in file]
    validate = np.random.permutation(validate)
    # import ipdb; ipdb.set_trace()
    if num:
        validate = validate[:num]
    for filename in validate:
        # import ipdb; ipdb.set_trace()
        path = working_dir + dataset_name + '/' + filename
        eval_env = HMTJerichoEnv(path, None, fully, step_limit=step_limit, get_valid=True)
        print(eval_env.json_path)

        obs, info = eval_env.reset()
        idx_start = obs.find('The human agent has')
        idx_end = obs.find('Now you are')
        obs = obs[:idx_start] + obs[idx_end:]
        obs = obs.replace('#', ' ')
        obs = obs.replace(',', '')
        done = False
        reward = 0
        agent.reset(eval_env.env)
        actions = list()
        for _ in range(step_limit):
            action = agent.act(obs, reward, done, info)
            actions.append(action)
            ob, reward, done, info = eval_env.step(action)
            # print(action, ob)
            ob = ob.replace('#', ' ')  # simplify: remove '#'
            ob = ob.replace(',', '')
            obs += ' [SEP] ' + action + ' [SEP] '
            obs += ob
            # import ipdb; ipdb.set_trace()
            if not fully:
                obs += eval_env.env.get_look(fully=False).replace('#', ' ')
            if done:
                break
        if reward > 0:
            print('Succeed!')
            print(actions)
        else:
            print('Fail!')
            if 'change_state' not in filename:
                print('wrong!')
                pass
            print(actions)
        eval_results.append((info['moves'], info['score'], info['question_cost']))
        print('moves: ', info['moves'], '; score: ', info['score'])
    print(eval_results)
    return eval_results


if __name__ == '__main__':
    np.random.seed(0)
    fully = True
    level = 'level1'
    agent = RandomAgent()
    eval_results = evaluate(agent, fully=fully, level=level)
    scores = [result[1] for result in eval_results]
    scores = np.array(scores)
    average = np.mean(scores)
    success = np.where(scores > 0)
    success_rate = float(len(success[0]) / len(scores))
    if success_rate != 0:
        average_score_when_success = np.mean(scores[success])
    else:
        average_score_when_success = 0.0

    print('random agent')
    print('fully:', fully)
    print('level:', level)
    print('Average Score:', average)
    print('Success Rate:', success_rate)
    print('Average Score On Success Cases:', average_score_when_success)


