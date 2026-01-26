import torch
import torch.nn as nn
import math
from GBMEnv_mv import globalsetting
from PPO_agent_v2 import JointBandPolicy, ValueNetCLS, PPOConfig

def ppo_update_joint(policy: JointBandPolicy, valuef: ValueNetCLS, opt_pi, opt_v, cfg: PPOConfig, batch,
                     env_cfg: globalsetting,stage:int=2, width_prior_w:float=0.02):
    obs = batch['obs']; m_pre = batch['m']; s_pre = batch['s']; logp_old = batch['logp']
    adv = batch['adv']; ret = batch['ret']
    N = obs.shape[0]; n_assets = m_pre.shape[1]

    for _ in range(cfg.epochs):
        idx = torch.randperm(N, device=obs.device)
        for start in range(0, N, cfg.minibatch_size):
            sel = idx[start:start+cfg.minibatch_size]
            o = obs[sel]; mp = m_pre[sel].detach(); sp = s_pre[sel].detach()
            lp_old = logp_old[sel].detach(); adv_o = adv[sel].detach(); ret_o = ret[sel].detach()

            if stage == 2:
                lp_new = policy.logprob_s_only(o, sp)
            else:
                lp_new = policy.logprob_m_only(o, mp)
            
            ratio = torch.exp(lp_new - lp_old)
            surr1 = ratio * adv_o
            surr2 = torch.clamp(ratio, 1.0-cfg.clip_ratio, 1.0+cfg.clip_ratio) * adv_o

            # エントロピー（mは凍結中でも項は一定。s の探索源）
            if stage == 1:
                ent = n_assets * (0.5*(1.0+math.log(2*math.pi)) + policy.log_std_m)  # mだけ
            else:
                ent = n_assets * (0.5*(1.0+math.log(2*math.pi)) + policy.log_std_s)  # sだけ
            loss_pi = -torch.min(surr1, surr2).mean() - cfg.entropy_coef * ent

            # --- λ→幅の弱教師（Stage-2のみ）
            if stage == 2 and width_prior_w > 0.0:
                # 観測の global_feats: [lam, mean_sigma, port_var, ||R w||]
                lam = o[:, (o.shape[1]-4)].unsqueeze(1)             # [B,1]
                g_star = ((1.0 - lam).clamp(min=0.0) + 1e-8).pow(env_cfg.ALPHA)  # [B,1]
                # s_pre は (0,1) の“射影前のシグモイド結果”。そのまま比較でOK
                loss_width = ((sp - g_star.expand_as(sp))**2).mean()
                loss_pi = loss_pi + width_prior_w * loss_width

            # simplex (弱め)
            excess = torch.clamp(mp.sum(dim=1) - 1.0, min=0.0)
            loss_pi = loss_pi + 1e-3 * (excess**2).mean()

            if opt_pi is not None:
                opt_pi.zero_grad()
                loss_pi.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                opt_pi.step()

            v_pred = valuef(o)
            loss_v = 0.5 * (v_pred - ret_o).pow(2).mean() * cfg.vf_coef
            opt_v.zero_grad(); loss_v.backward()
            nn.utils.clip_grad_norm_(valuef.parameters(), cfg.max_grad_norm); opt_v.step()