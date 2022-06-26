# Built-in imports
import argparse
import os.path
import random
import logging

# Third party imports
import jericho
import torch
import numpy as np


# Custom imports
from handmethat.model.DRRN.drrn_agent import DrrnAgent
from handmethat.model.DRRN.drrn_trainer import DrrnTrainer
from handmethat.model.offlineDRRN.offline_drrn_agent import offlineDrrnAgent
from handmethat.model.offlineDRRN.offline_drrn_trainer import offlineDrrnTrainer

import definitions.defs as defs
from handmethat.envs.jericho_env import HMTJerichoEnv
from utils.vec_env import VecEnv
from utils import logger
from utils.memory import State, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.getLogger().setLevel(logging.CRITICAL)


# torch.autograd.set_detect_anomaly(True)


def configure_logger(args):
    log_dir = args.output_dir
    type_strs = ["json", "stdout"]
    tb = logger.Logger(
        log_dir,
        [
            logger.make_output_format(type_str, log_dir, args=args)
            for type_str in type_strs
        ],
    )

    logger.configure("{}/{}-{}-{}-{}".format(log_dir, args.model, args.observability, args.eval_split, args.level),
                     format_strs=["log", "stdout"], off=args.logging_off)
    log = logger.log

    return tb, log


def parse_args():
    parser = argparse.ArgumentParser()

    # General Settings
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--data_dir_name', default='HandMeThat_with_expert_demonstration')
    parser.add_argument('--save_path', default='./models')
    parser.add_argument('--logging_off', default=0, type=int)
    parser.add_argument('--weight_file', default=None, type=str)
    parser.add_argument('--memory_file', default=None, type=str)
    parser.add_argument('--project_name', default='HandMeThat', type=str)
    parser.add_argument('--debug', default=0, type=int)

    # not needed
    parser.add_argument('--rom_path', default='data', type=str)
    parser.add_argument('--traj_file', default=None, type=str)
    parser.add_argument('--run_id', default=None, type=str)
    parser.add_argument('--wandb', default=1, type=int)
    parser.add_argument('--jericho_add_wt', default='add_wt', type=str)

    # Environment settings
    parser.add_argument('--check_valid_actions_changed', default=0, type=int)

    # Training Settings
    parser.add_argument('--env_step_limit', default=40, type=int)
    parser.add_argument('--dynamic_episode_length', default=0, type=int)
    parser.add_argument('--episode_ext_type', default='steady_50', type=str)
    parser.add_argument('--num_envs', default=8, type=int)              # number of envs per step
    parser.add_argument('--max_steps', default=100000, type=int)        # set max training steps
    parser.add_argument('--q_update_freq', default=1, type=int)         # update model
    parser.add_argument('--checkpoint_freq', default=5000, type=int)    # save model
    parser.add_argument('--eval_freq', default=5000, type=int)          # (useless, not implemented)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--target_update_freq', default=100, type=int)
    parser.add_argument('--dump_traj_freq', default=5000, type=int)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--memory_size', default=500000, type=int)
    parser.add_argument('--memory_alpha', default=.4, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--priority_fraction', default=0.5, type=float)
    parser.add_argument('--no_invalid_act_detect', default=0, type=int)
    parser.add_argument('--filter_invalid_acts', default=1, type=int)
    parser.add_argument('--start_from_reward', default=0, type=int)
    parser.add_argument('--start_from_wt', default=0, type=int)
    parser.add_argument('--filter_drop_acts', default=0, type=int)

    # Action Model Settings
    parser.add_argument('--max_acts', default=5, type=int)
    parser.add_argument('--tf_embedding_dim', default=128, type=int)
    parser.add_argument('--tf_hidden_dim', default=128, type=int)
    parser.add_argument('--nhead', default=4, type=int)
    parser.add_argument('--feedforward_dim', default=512, type=int)
    parser.add_argument('--tf_num_layers', default=3, type=int)
    parser.add_argument('--ngram', default=3, type=int)
    parser.add_argument('--traj_k', default=1, type=int)
    parser.add_argument('--action_model_update_freq', default=1e9, type=int)
    parser.add_argument('--smooth_alpha', default=0.00001, type=float)
    parser.add_argument('--cut_beta_at_threshold', default=0, type=int)
    parser.add_argument('--action_model_type', default='ngram', type=str)
    parser.add_argument('--tf_num_epochs', default=50, type=int)
    parser.add_argument(
        '--turn_action_model_off_after_falling', default=0, type=int)
    parser.add_argument('--traj_dropout_prob', default=0, type=float)
    parser.add_argument('--init_bin_prob', default=0.1, type=float)
    parser.add_argument('--num_bins', default=0, type=int)
    parser.add_argument('--binning_prob_update_freq', default=1e9, type=int)
    parser.add_argument('--random_action_dropout', default=0, type=int)
    parser.add_argument('--use_multi_ngram', default=0, type=int)
    parser.add_argument('--use_action_model', default=0, type=int)
    parser.add_argument('--sample_action_argmax', default=0, type=int)
    parser.add_argument('--il_max_context', default=512, type=int)
    parser.add_argument('--il_k', default=5, type=int)
    parser.add_argument('--il_batch_size', default=64, type=int)
    parser.add_argument('--il_lr', default=1e-3, type=float)
    parser.add_argument('--il_max_num_epochs', default=200, type=int)
    parser.add_argument('--il_num_eval_runs', default=3, type=int)
    parser.add_argument('--il_eval_freq', default=300, type=int)
    parser.add_argument('--il_vocab_size', default=2000, type=int)
    parser.add_argument('--il_temp', default=1., type=float)
    parser.add_argument('--use_il', default=0, type=int)
    parser.add_argument('--il_len_scale', default=1.0, type=float)
    parser.add_argument('--use_il_graph_sampler', default=0, type=int)
    parser.add_argument('--use_il_buffer_sampler', default=1, type=int)
    parser.add_argument('--il_top_p', default=0.9, type=float)
    parser.add_argument('--il_use_dropout', default=0, type=int)
    parser.add_argument('--il_use_only_dropout', default=0, type=int)

    # DRRN Model Settings
    parser.add_argument('--drrn_embedding_dim', default=128, type=int)
    parser.add_argument('--drrn_hidden_dim', default=128, type=int)
    parser.add_argument('--use_drrn_inv_look', default=1, type=int)
    parser.add_argument('--use_counts', default=0, type=int)
    parser.add_argument('--reset_counts_every_epoch', default=0, type=int)
    parser.add_argument('--sample_uniform', default=0, type=int)
    parser.add_argument('--T', default=1, type=float)
    parser.add_argument('--rotating_temp', default=0, type=int)
    parser.add_argument('--augment_state_with_score', default=0, type=int)

    # Graph Model Settings
    parser.add_argument('--graph_num_explore_steps', default=20, type=int)
    parser.add_argument('--graph_rescore_freq', default=500, type=int)
    parser.add_argument('--graph_merge_freq', default=500, type=int)
    parser.add_argument('--graph_hash', default='inv_loc_ob', type=str)
    parser.add_argument('--graph_score_temp', default=1, type=float)
    parser.add_argument('--graph_q_temp', default=1, type=float)
    parser.add_argument('--graph_alpha', default=0.5, type=float)
    parser.add_argument('--log_top_blue_acts_freq', default=100, type=int)

    # Offline Q Learning settings
    parser.add_argument('--offline_q_steps', default=1000, type=int)
    parser.add_argument('--offline_q_transfer_freq', default=100, type=int)
    parser.add_argument('--offline_q_eval_runs', default=10, type=int)

    # Inv-Dyn Settings
    parser.add_argument('--type_inv', default='decode')
    parser.add_argument('--type_for', default='ce')
    parser.add_argument('--w_inv', default=0, type=float)
    parser.add_argument('--w_for', default=0, type=float)
    parser.add_argument('--w_act', default=0, type=float)
    parser.add_argument('--r_for', default=0, type=float)

    parser.add_argument('--nor', default=0, type=int, help='no game reward')
    parser.add_argument('--randr', default=0, type=int,
                        help='random game reward by objects and locations within episode')
    parser.add_argument('--perturb', default=0, type=int,
                        help='perturb state and action')

    parser.add_argument('--hash_rep', default=0, type=int,
                        help='hash for representation')
    parser.add_argument('--act_obs', default=0, type=int,
                        help='action set as state representation')
    parser.add_argument('--fix_rep', default=0, type=int,
                        help='fix representation')

    # Additional Model Settings
    parser.add_argument('--model_name', default='drrn', type=str)
    parser.add_argument('--beta', default=0.3, type=float)
    parser.add_argument('--beta_trainable', default=0, type=int)
    parser.add_argument(
        '--eps',
        default=0,
        type=int,
        help='0: ~ softmax act_value; 1: eps-greedy-exploration',
    )
    parser.add_argument(
        '--eps_type',
        default='uniform',
        type=str,
        help='uniform (-1): uniform exploration; softmax_lm (0): ~ softmax lm_value; uniform_lm_topk (>0): ~ uniform(top k w.r.t. lm_value)',
    )
    parser.add_argument(
        '--alpha',
        default=0,
        type=float,
        help='act_value = alpha * bert_value + (1-alpha) * q_value; only used when eps is None now',
    )
    parser.add_argument('--sample_argmax',
                        default=0,
                        type=int,
                        help='whether to replace sampling with argmax')

    # todo: HandMeThat arguments
    parser.add_argument('--observability', default='fully', type=str)   # fully or partially
    parser.add_argument('--level', default='level1', type=str)               # hardness level 1,2,3,4 (for eval)
    parser.add_argument('--eval_split', default='test', type=str)       # test or validate

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--jericho_seed', default=0, type=int)
    parser.add_argument('--model', default='DRRN', type=str)            # DRRN or offlineDRRN

    return parser.parse_args()


def main():
    assert jericho.__version__.startswith(
        "3"), "This code is designed to be run with Jericho version >= 3.0.0."
    args = parse_args()
    print(args)
    print("device", device)
    print(args.model)

    # Set seed across imports
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Start logger
    tb, log = configure_logger(args)

    if args.debug:
        import pdb
        pdb.set_trace()


    # Setup envs
    cache = dict()
    if args.observability == 'fully':
        fully = True
    elif args.observability == 'partially':
        fully = False
    else:
        raise Exception('Unknown observability!')
    eval_env = HMTJerichoEnv(args.data_path + '/' + args.data_dir_name,
                             args.eval_split,
                             fully=fully,
                             step_limit=args.env_step_limit,
                             get_valid=True,
                             seed=args.jericho_seed,
                             args=args,
                             cache=cache,
                             start_from_reward=args.start_from_reward,
                             start_from_wt=args.start_from_wt,
                             log=log)

    # Setup rl model
    if args.model_name == defs.DRRN:
        assert args.use_action_model == 0, "'use_action_model' needs to be OFF"
        assert args.r_for == 0, "r_for needs to be zero when NOT using inverse dynamics."
        assert args.use_il == 0, "no il should be used when running DRRN."
        if args.model == 'DRRN':
            agent = DrrnAgent(tb, log, args, None, None)
            trainer = DrrnTrainer(tb, log, agent, None, eval_env, args)     # omit the eval_env
        elif args.model == 'offlineDRRN':
            agent = offlineDrrnAgent(tb, log, args, None, None)
            trainer = offlineDrrnTrainer(tb, log, agent, None, eval_env, args)  # omit the eval_env
        else:
            raise Exception("Unknown model type!")
    else:
        raise Exception("Unknown model type!")

    MODEL_DIR = os.path.join(args.save_path, args.model, args.observability)

    if args.weight_file is not None and args.memory_file is not None:
        agent.load(args.weight_file, args.memory_file, MODEL_DIR)
        log("Successfully loaded network and replay buffer from checkpoint!")

    trainer.eval(fully=fully, level=args.level)


if __name__ == "__main__":
    main()
