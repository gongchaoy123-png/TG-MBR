"""TARGET: target-guided multi-granularity auxiliary refinement."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Params import args


class TARGET(nn.Module):
    def __init__(self, num_user, num_item, adj_list):
        super().__init__()
        self.num_user = num_user
        self.latdim = args.hidden_dim
        self.gcn_layer = args.gnn_layer
        self.behavior_num = len(adj_list)
        self.lam3 = args.reg
        self.adj_list = adj_list
        self.current_epoch = 0
        self.alpha_logits = (
            nn.Parameter(torch.zeros(self.behavior_num))
            if args.behavior_fusion in ('importance', 'attention') else None
        )

        self.user_emb = nn.Parameter(torch.randn(num_user, self.latdim) * 0.1)
        self.item_emb = nn.Parameter(torch.randn(num_item, self.latdim) * 0.1)
        self.W_u = nn.Parameter(torch.eye(self.latdim))
        self.W_i = nn.Parameter(torch.eye(self.latdim))
        self.W_l = nn.Parameter(torch.cat([
            torch.eye(self.latdim) for _ in range(self.gcn_layer)
        ], dim=0))
        self.prelu = nn.PReLU()

        auxiliary_count = self.behavior_num - 1
        self.use_interaction_filter = (
            args.interaction_filter and args.interaction_filter_strength > 0 and auxiliary_count > 0
        )
        if self.use_interaction_filter:
            self.behavior_context = nn.Parameter(
                torch.randn(auxiliary_count, self.latdim) * 0.02
            )
            self.interaction_pre_filters = nn.ModuleList([
                nn.Linear(self.latdim * 5, self.latdim)
                for _ in range(auxiliary_count)
            ])
            # Candidate-aware auxiliary preference mapping. Keep the parameter key
            # interaction_item_layers for compatibility with existing checkpoints.
            self.interaction_item_layers = nn.ModuleList([
                nn.Linear(self.latdim * 3, self.latdim)
                for _ in range(auxiliary_count)
            ])
            self.interaction_post_filters = nn.ModuleList([
                nn.Linear(self.latdim * 5, self.latdim)
                for _ in range(auxiliary_count)
            ])
            # Per-(user,item,behavior) skip gates.  The gate decides whether the
            # filtered interaction residual should be used at all.  Zero final
            # weights and a negative bias make the initial model close to the
            # original, unfiltered scorer.
            self.interaction_skip_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.latdim * 7, self.latdim), nn.ReLU(),
                    nn.Linear(self.latdim, 1)
                )
                for _ in range(auxiliary_count)
            ])
            for group in (self.interaction_pre_filters,
                          self.interaction_item_layers,
                          self.interaction_post_filters):
                for layer in group:
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
            for gate in self.interaction_skip_gates:
                nn.init.xavier_uniform_(gate[0].weight)
                nn.init.zeros_(gate[0].bias)
                nn.init.zeros_(gate[2].weight)
                nn.init.constant_(gate[2].bias, -2.0)
        self.use_user_denoising = args.denoise_mode in ('user', 'both') and auxiliary_count > 0
        self.use_item_denoising = args.denoise_mode in ('item', 'both') and auxiliary_count > 0
        self.use_denoising = self.use_user_denoising or self.use_item_denoising
        if self.use_denoising:
            self.user_denoise_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.latdim * 3, self.latdim), nn.ReLU(),
                    nn.Linear(self.latdim, 1))
                for _ in range(auxiliary_count if self.use_user_denoising else 0)
            ])
            self.item_denoise_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.latdim * 3, self.latdim), nn.ReLU(),
                    nn.Linear(self.latdim, 1))
                for _ in range(auxiliary_count if self.use_item_denoising else 0)
            ])
            for gate_group in (self.user_denoise_gates, self.item_denoise_gates):
                for gate in gate_group:
                    nn.init.xavier_uniform_(gate[0].weight)
                    nn.init.zeros_(gate[0].bias)
                    # Start close to pass-through: denoising should be learned
                    # rather than destroy the stable baseline at initialization.
                    nn.init.zeros_(gate[2].weight)
                    nn.init.constant_(gate[2].bias, 2.0)

        print(f'TARGET: behavior fusion={args.behavior_fusion}, '
              f'K={self.behavior_num}, L={self.gcn_layer}, d={self.latdim}, '
              f'denoising={args.denoise_mode}')

    def _propagate(self, adj):
        ego = torch.cat([self.user_emb, self.item_emb], dim=0)
        layers = [ego]
        for _ in range(self.gcn_layer):
            ego = torch.sparse.mm(adj, ego)
            layers.append(ego)
        return layers

    def _behavior_weights(self):
        if self.alpha_logits is None:
            return self.user_emb.new_full((self.behavior_num,), 1.0 / self.behavior_num)
        return torch.softmax(self.alpha_logits, dim=0)

    def _denoise_layers(self, layers, gates):
        auxiliary = layers[:, :-1]
        target = layers[:, -1:]
        target_query = target.expand_as(auxiliary)
        gate_input = torch.cat([auxiliary, target_query, auxiliary - target_query], dim=-1)
        floor = min(max(float(args.denoise_gate_floor), 0.0), 1.0)
        weights = torch.stack([
            floor + (1.0 - floor) * torch.sigmoid(gate(gate_input[:, index]))
            for index, gate in enumerate(gates)
        ], dim=1)
        return torch.cat([weights * auxiliary, target], dim=1)

    def forward(self):
        K, L = self.behavior_num, self.gcn_layer
        layers = [self._propagate(adj) for adj in self.adj_list]
        # Keep layer representations separate so auxiliary noise can be
        # filtered before layer fusion instead of only after the final layer.
        layer_u = torch.stack([
            torch.stack([
                layers[k][l][:self.num_user] @ self.W_u
                for l in range(1, L + 1)
            ], dim=1)
            for k in range(K)
        ], dim=1)  # [U, K, L, d]
        layer_i = torch.stack([
            torch.stack([
                layers[k][l][self.num_user:] @ self.W_i
                for l in range(1, L + 1)
            ], dim=1)
            for k in range(K)
        ], dim=1)  # [I, K, L, d]

        if self.use_user_denoising:
            layer_u = self._denoise_layers(layer_u, self.user_denoise_gates)
        if self.use_item_denoising:
            layer_i = self._denoise_layers(layer_i, self.item_denoise_gates)

        # Fuse the denoised layer views into one representation per behavior.
        behavior_u = self.prelu(
            layer_u.reshape(
                layer_u.shape[0], K, L * self.latdim
            ) @ self.W_l
        )
        behavior_i = self.prelu(
            layer_i.reshape(
                layer_i.shape[0], K, L * self.latdim
            ) @ self.W_l
        )
        behavior_weight = self._behavior_weights().view(1, K, 1)
        v_u = (behavior_u * behavior_weight).sum(dim=1)
        v_i = (behavior_i * behavior_weight).sum(dim=1)
        return v_u, v_i, behavior_u, behavior_i

    def interaction_target_scores(self, users, items, behavior_u, behavior_i,
                                  behavior_weight):
        """Score pairs using target-guided filtering and candidate-aware auxiliary preference mapping."""
        target_u = behavior_u[users, -1]
        target_i = behavior_i[items, -1]
        residuals = []
        for b, skip_filter in enumerate(self.interaction_skip_gates):
            u_b = behavior_u[users, b]
            i_b = behavior_i[items, b]
            b_ctx = self.behavior_context[b].unsqueeze(0).expand_as(u_b)
            triple = torch.cat([u_b, i_b, b_ctx, target_u, target_i], dim=-1)
            pre_gate = torch.sigmoid(self.interaction_pre_filters[b](triple))
            filtered_u = pre_gate * u_b
            pref = torch.tanh(self.interaction_item_layers[b](
                torch.cat([filtered_u, i_b, b_ctx], dim=-1)))
            post_gate = torch.sigmoid(self.interaction_post_filters[b](triple))
            pref = post_gate * pref
            filtered_score = (pref * i_b).sum(dim=-1)
            base_score = (u_b * i_b).sum(dim=-1)
            gate_input = torch.cat([
                u_b, i_b, b_ctx, target_u, target_i, u_b - target_u, i_b - target_i
            ], dim=-1)
            skip_gate = torch.sigmoid(skip_filter(gate_input).squeeze(-1))
            residuals.append(behavior_weight[b] * skip_gate * (filtered_score - base_score))
        return torch.stack(residuals, dim=0).sum(dim=0)

    def _interaction_strength(self):
        strength = min(max(float(args.interaction_filter_strength), 0.0), 1.0)
        warmup = max(int(args.interaction_filter_warmup), 0)
        ramp = max(int(args.interaction_filter_ramp), 1)
        progress = (float(self.current_epoch) - warmup) / ramp
        return strength * min(max(progress, 0.0), 1.0)

    @torch.no_grad()
    def predict_target_scores(self, users, items, return_base=False):
        return self.score_encoded_candidates(users, items, self.forward(), return_base)

    def score_encoded_candidates(self, users, items, encoded, return_base=False):
        v_u, v_i, behavior_u, behavior_i = encoded
        base = (v_u[users] * v_i[items]).sum(dim=-1)
        strength = self._interaction_strength() if self.use_interaction_filter else 0.0
        if strength == 0.0:
            return (base, base) if return_base else base
        behavior_weight = self._behavior_weights()
        filtered_residual = self.interaction_target_scores(
            users, items, behavior_u, behavior_i, behavior_weight)
        score = base + strength * filtered_residual
        return (score, base) if return_base else score

    def bpr_multi_loss(self, v_u, v_i, users, pos, neg, behavior_u, behavior_i):
        loss, valid = None, 0
        for k in range(self.behavior_num):
            mask = pos[:, k] >= 0
            if mask.sum() == 0:
                continue
            u = v_u[users[mask]]
            p = v_i[pos[:, k][mask]]
            n = v_i[neg[:, k][mask]]
            if k == self.behavior_num - 1:
                batch_users = users[mask]
                pos_items = pos[:, k][mask]
                neg_items = neg[:, k][mask]
                base_score = (u * p).sum(-1) - (u * n).sum(-1)
                mixed_score = base_score
                strength = self._interaction_strength() if self.use_interaction_filter else 0.0
                if strength:
                    behavior_weight = self._behavior_weights()
                    pos_residual = self.interaction_target_scores(
                        batch_users, pos_items, behavior_u, behavior_i,
                        behavior_weight)
                    neg_residual = self.interaction_target_scores(
                        batch_users, neg_items, behavior_u, behavior_i,
                        behavior_weight)
                    mixed_score = mixed_score + strength * (
                        pos_residual - neg_residual)
                current = -F.logsigmoid(mixed_score).mean()
            else:
                current = -F.logsigmoid(
                    (u * p).sum(-1) - (u * n).sum(-1)
                ).mean()
            loss = current if loss is None else loss + current
            valid += 1
        if loss is None:
            return v_u.sum() * 0.0
        return loss / valid

    def cal_loss(self, users, pos, neg):
        v_u, v_i, behavior_u, behavior_i = self.forward()
        bpr = self.bpr_multi_loss(v_u, v_i, users, pos, neg, behavior_u, behavior_i)
        mask = pos >= 0
        used_users = users.unsqueeze(1).expand_as(pos)[mask]
        reg = ((v_u[used_users] ** 2).sum() +
               (v_i[pos[mask]] ** 2).sum() +
               (v_i[neg[mask]] ** 2).sum()) / max(int(mask.sum()), 1)
        return bpr + self.lam3 * reg, bpr.detach(), reg.detach()
