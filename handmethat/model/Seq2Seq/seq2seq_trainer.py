import datetime
import os
import json
import importlib
import numpy as np
import torch.cuda
from tqdm import tqdm

import sys

import alfworld.agents.environment
# import alfworld.agents.modules.generic as generic
from alfworld.agents.agent import TextDAggerAgent
from alfworld.agents.eval import evaluate_dagger, my_evaluate
from alfworld.agents.modules.generic import HistoryScoreCache, EpisodicCountingMemory, ObjCentricEpisodicMemory

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(config, args):
    # import ipdb; ipdb.set_trace()

    time_1 = datetime.datetime.now()
    agent = TextDAggerAgent(config)
    MODEL_DIR = os.path.join(args.save_path, args.model, args.observability)
    MAX_TRAIN_STEP = args.max_train_step
    REPORT_FREQUENCY = args.report_frequency

    episode_no = 0
    running_avg_dagger_loss = HistoryScoreCache(capacity=500)

    agent.load_from_tag = args.load_from_tag
    agent.load_pretrained = bool(args.load_pretrained)
    if agent.load_pretrained:
        if os.path.exists(MODEL_DIR + "/" + agent.load_from_tag + ".pt"):
            agent.load_pretrained_model(MODEL_DIR + "/" + agent.load_from_tag + ".pt")
            agent.update_target_net()
            episode_no = int(agent.load_from_tag.split('_')[-1])
    # load dataset
    # push experience into replay buffer (dagger)
    data_info = args.data_path + '/HandMeThat_data_info.json'
    if args.observability == 'fully':
        fully = True
    elif args.observability == 'partially':
        fully = False
    else:
        raise Exception('Unknown observability!')

    with open(data_info, 'r') as f:
        json_str = json.load(f)
    train_files = json_str[2]['train']
    train_files = np.random.permutation(train_files)    # random permutation for input
    # import ipdb; ipdb.set_trace()
    for file in train_files:
        with open(args.data_path + '/' + args.data_dir_name + '/' + file, 'r') as f:
            json_str = json.load(f)
            actions = json_str['demo_actions']
            if fully:
                observations = json_str['demo_observations_fully']
            else:
                observations = json_str['demo_observations_partially']
            task = json_str['task_description']
            trajectory = []
            for i in range(len(actions)):
                obs = observations[i]
                obs_list = obs.split()
                if len(obs_list) > 1000:
                    obs_list = obs_list[:500] + obs_list[-500:]     # avoid too long obs
                obs = ' '.join(obs_list)
                action = actions[i]
                trajectory.append([obs, task, None, action, None])
            agent.dagger_memory.push(trajectory)
    # import ipdb; ipdb.set_trace()
    while True:
        print(episode_no, datetime.datetime.now())
        # import ipdb; ipdb.set_trace()
        if episode_no > MAX_TRAIN_STEP:
            break

        agent.train()
        for i in range(4):
            dagger_loss = agent.update_dagger()
            print('dagger loss:', dagger_loss)
            if dagger_loss is not None:
                running_avg_dagger_loss.push(dagger_loss)

        report = (episode_no % REPORT_FREQUENCY == 0 and episode_no > 0)
        episode_no += 10
        if not report:
            continue
        time_2 = datetime.datetime.now()
        print("Episode: {:3d} | time spent: {:s} | loss: {:2.3f}".format(episode_no, str(time_2 - time_1).rsplit(".")[0], running_avg_dagger_loss.get_avg()))

        model_name = 'weights_{}.pt'.format(episode_no - 10)
        agent.save_model_to_path(MODEL_DIR + '/' + model_name)
