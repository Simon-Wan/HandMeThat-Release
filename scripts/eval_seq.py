import copy
import torch

import os
import sys
import json
import argparse
import yaml

from alfworld.agents.utils.misc import extract_admissible_commands
from handmethat.model.Seq2Seq.seq2seq_agent import Seq2SeqAgent
from handmethat.envs.jericho_env import HMTJerichoEnv
import numpy as np

import alfworld.agents.modules.generic as generic
from alfworld.agents.agent import TextDAggerAgent


def load_config():
    '''from alfworld.agents.modules.generic'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default='./scripts/seq_config.yaml', help="path to config file")
    parser.add_argument("-p", "--params", nargs="+", metavar="my.setting=value", default=[],
                        help="override params of the config file,"
                             " e.g. -p 'training.gamma=0.95'")

    # add for HandMeThat
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--data_dir_name', default='HandMeThat_with_expert_demonstration')
    parser.add_argument('--save_path', default='./models')
    parser.add_argument('--observability', default='fully', type=str)
    parser.add_argument('--load_pretrained', default=0, type=int)
    parser.add_argument('--load_from_tag', default=None, type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model', default='Seq2Seq', type=str)
    parser.add_argument('--max_train_step', default=100000, type=int)
    parser.add_argument('--report_frequency', default=5000, type=int)

    parser.add_argument('--level', default='level1', type=str)
    parser.add_argument('--eval_split', default='test', type=str)
    parser.add_argument('--eval_model_name', default=None, type=str)

    args = parser.parse_args()
    assert os.path.exists(args.config_file), "Invalid config file"
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    # Parse overriden params.
    for param in args.params:
        fqn_key, value = param.split("=")
        entry_to_change = config
        keys = fqn_key.split(".")
        for k in keys[:-1]:
            entry_to_change = entry_to_change[k]
        entry_to_change[keys[-1]] = yaml.load(value)
    # print(config)
    return config, args


def main():
    config, args = load_config()
    agent = TextDAggerAgent(config)
    model_name = args.eval_model_name
    MODEL_DIR = os.path.join(args.save_path, args.model, args.observability)
    if os.path.exists(MODEL_DIR + '/' + model_name):
        agent.load_pretrained_model(MODEL_DIR + '/' + model_name)
        agent.update_target_net()
    if args.observability == 'fully':
        fully = True
    elif args.observability == 'partially':
        fully = False
    else:
        raise Exception('Unknown observability!')
    level = args.level
    agent.eval()
    my_agent = Seq2SeqAgent(agent)
    with torch.no_grad():
        eval_results = evaluate_for_seq2seq(args, my_agent, fully=fully, level=level)
    scores = [result[1] for result in eval_results]
    scores = np.array([scores])
    average_score = np.mean(scores)
    success = np.where(scores > 0)
    success_rate = float(len(success[0]) / len(scores[0]))
    if success_rate != 0:
        average_score_when_success = np.mean(scores[success])
    else:
        average_score_when_success = 0.0
    results = {
        'average_score': average_score,
        'success_rate': success_rate,
        'average_score_when_success': average_score_when_success,
    }
    print('{}, {}, {}'.format(args.model, args.level, args.observability))
    for key in results.keys():
        print(key, results[key])


def evaluate_for_seq2seq(args, agent, fully, level=None):
    eval_results = list()
    step_limit = 40
    data_path = args.data_path
    data_dir_name = args.data_dir_name
    data_info = data_path + '/HandMeThat_data_info.json'
    with open(data_info, 'r') as f:
        json_str = json.load(f)
    eval_files = json_str[2][args.eval_split]
    if level:
        eval_files = [file for file in eval_files if level in file]
    eval_files = np.random.permutation(eval_files)
    for filename in eval_files:
        path = os.path.join(data_path, data_dir_name, filename)
        eval_env = HMTJerichoEnv(path, None, fully, step_limit=step_limit, get_valid=True)
        print(eval_env.json_path)

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
            print('Action:', action, '|| Observation:', ob)
            ob = ob.replace('#', ' ')
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
        print('moves: ', info['moves'], '; score: ', info['score'], '\n')
    return eval_results


if __name__ == '__main__':
    main()
