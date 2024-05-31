from typing import Optional, Union
from uu import decode

from einops import rearrange
import torch
import torch.nn as nn

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.models.common.constructive.nonautoregressive import (
    NonAutoregressiveEncoder,
    NonAutoregressiveDecoder,
    NonAutoregressivePolicy,
)
from rl4co.models.zoo.nargnn.encoder import NARGNNEncoder
from rl4co.utils.ops import batchify, gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict

from GLOP.heatmap.cvrp.eval import eval

log = get_pylogger(__name__)


class GLOPOriPolicy(NonAutoregressivePolicy):
    def __init__(
        self,
        encoder: NonAutoregressiveEncoder = None,
        decoder: NonAutoregressiveDecoder = None,
        env_name: Union[str, RL4COEnvBase] = "tsp",
        n_samples: int = 10,
        opts: list[Union[callable]] = None,
        **encoder_kwargs,
    ):
        if encoder is None:
            encoder = NARGNNEncoder(**encoder_kwargs)
        if decoder is None:
            decoder = NonAutoregressiveDecoder()

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            train_decode_type="multistart_sampling",
            val_decode_type="multistart_sampling",
            test_decode_type="multistart_sampling",
        )

        self.n_samples = n_samples
        self.opts = opts

    def forward(
        self,
        td: TensorDict,
        env: Union[str, RL4COEnvBase, None] = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = False,
        return_entropy: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        return_partitions: bool = True,
        return_partitions_actions: bool = True,
        actions=None,
        **decoding_kwargs,
    ) -> dict:
        device = td.device
        
        par_out = super().forward(
            td = td,
            env = env,
            phase = phase,
            calc_reward = False, # We don't need the partition reward
            return_actions = True, # Used for partition
            return_entropy = return_entropy,
            return_init_embeds = return_init_embeds,
            return_sum_log_likelihood = return_sum_log_likelihood,
            num_starts = self.n_samples,
            actions = actions,
            decode_type="multistart_sampling",
            **decoding_kwargs,
        )

        td_sample = batchify(td, self.n_samples)
        par_actions = par_out["actions"]
        par_log_likelihood = par_out["log_likelihood"]

        # Based on partition actions to get partitions
        tsp_insts_list, n_tsps_per_route_list = self.partition_glop(td, par_actions, self.n_samples)

        reward_list = []
        for batch_idx in range(td.batch_size[0]):
            tsp_insts = tsp_insts_list[batch_idx]
            n_tsps_per_route = n_tsps_per_route_list[batch_idx]
            objs = eval(tsp_insts, n_tsps_per_route, self.opts)
            reward_list.append(objs)

        reward = torch.stack(reward_list, dim=0)

        # Construct final output
        out = {"log_likelihood": par_log_likelihood, "reward": reward}

        return out

    @staticmethod
    @torch.no_grad()
    def partition_glop(td: TensorDict, actions: torch.Tensor, n_samples: int):
        """Partition based on the partition actions, from original GLOP
        Args:
            td [bs]: NOTE: different with our partition, this doesn't to be sampled
            actions [bs*n_samples, seq_len]
        Returns:
            tsp_insts_list [bs]: list of tsp instances, each has the size of [sum_num_tsps_of_samples, max_tsp_len, 2]
            n_tsps_per_route_list [bs[n_samples]]: list of number of tsps per route, each element is a list[int]
        """
        from GLOP.heatmap.cvrp.inst import trans_tsp

        batch_size = td.batch_size[0]
        tsp_insts_list = []
        n_tsps_per_route_list = []
        for batch_idx in range(batch_size):
            coors = td["locs"][batch_idx]
            routes = actions[(batch_idx * n_samples):((batch_idx + 1) * n_samples)]

            tsp_insts, n_tsps_per_route = trans_tsp(coors, routes)
            tsp_insts_list.append(tsp_insts)
            n_tsps_per_route_list.append(n_tsps_per_route)

        return tsp_insts_list, n_tsps_per_route_list
