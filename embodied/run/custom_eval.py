from collections import defaultdict
from functools import partial as bind

import elements
import embodied
import numpy as np

import os
from PIL import Image

from typing import List, Optional
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt

import random

import datetime

outdir = os.path.join("./", "debug_frames", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(outdir, exist_ok=True)

frame_idx = 0

def compute_MDS(embeddings: List[np.ndarray], similarity_metric: str = "euclidean",
                title: Optional[str] = "DreamerV3 Embeddings", save_path: Optional[str] = None):
    X = np.asarray(embeddings, dtype=np.float64)

    # Pairwise dissimilarities in original space
    D = pairwise_distances(X, metric=similarity_metric)

    metric = True
    normalized_stress = False

    print(f"metric: {metric}, similarity_measure: {similarity_metric}, normalized: {normalized_stress}")

    mds = MDS(
        n_components=2,
        metric=metric,                 # metric MDS
        dissimilarity="precomputed", # using precomputed distance matrix D
        n_init=8,                    # multiple restarts
        max_iter=1000,
        random_state=0,
        normalized_stress=normalized_stress,
    )

    X_transformed = mds.fit_transform(D)  # shape (n, 2)

    # --- Preservation metrics ---
    D_2d = squareform(pdist(X_transformed, metric="euclidean"))  # pairwise distances in 2D
    mask = np.triu_indices_from(D, k=1)                          # unique pairs only

    pearson_r = pearsonr(D[mask], D_2d[mask]).statistic
    spearman_rho = spearmanr(D[mask], D_2d[mask]).statistic
    rmse_distance = np.sqrt(np.mean((D[mask] - D_2d[mask]) ** 2))

    print(f"Stress: {mds.stress_}")
    print(f"Pearson r (distance preservation): {pearson_r:.4f}")
    print(f"Spearman rho (rank preservation):  {spearman_rho:.4f}")
    # print(f"RMSE (distance error):             {rmse_distance:.4f}")

    # --- Plot ---
    plt.figure(figsize=(6, 6))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], s=30)

    # Highlight first and last points
    plt.scatter(
        X_transformed[0, 0], X_transformed[0, 1],
        s=100, color="green", alpha=0.9, zorder=6, label="Start"
    )
    plt.scatter(
        X_transformed[-1, 0], X_transformed[-1, 1],
        s=100, color="red", alpha=0.9, zorder=6, label="End"
    )

    for i, (x, y) in enumerate(X_transformed):
        plt.text(x, y, str(i), fontsize=8)

    if True:
        for i in range(len(X_transformed) - 1):
            x_start, y_start = X_transformed[i, 0], X_transformed[i, 1]
            x_end, y_end = X_transformed[i + 1, 0], X_transformed[i + 1, 1]
            dx = x_end - x_start
            dy = y_end - y_start
            # use annotate so arrow head size is specified in points (mutation_scale)
            plt.annotate(
                "",
                xy=(x_end, y_end),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle='->', lw=0.8, color='black', alpha=0.6, mutation_scale=10),
                zorder=4,
            )

    # --- Feature consistency ---
    con_mean, rand_val = feature_consistency(embeddings)
    fc_text = f"consecutive_frame_mean={con_mean:.3f}, random_frame_mean={rand_val:.3f}"
    # fc_text = ""

    plt.axis("equal")
    plt.title(f"{title}\nMetric MDS (2D) | metric={similarity_metric}\n{fc_text}")
    plt.legend()
    # plt.show()

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return X_transformed, {
        "stress": float(mds.stress_),
        "pearson_r": float(pearson_r),
        "spearman_rho": float(spearman_rho),
        "rmse_distance": float(rmse_distance),
    }

def feature_consistency(feat_list):
    n = len(feat_list)
    # grab two consecutive features. we expect them to be close
    consecutive_vals = []
    for i in range(n-1):
        sim = cosine_similarity(feat_list[i].reshape(1, -1), feat_list[i+1].reshape(1, -1))[0][0].item()
        consecutive_vals.append(sim)
    # grab two random features. we expect them to be far
    random_vals = []
    for i in range(n):
        idx1 = random.randint(0, n-1)
        idx2 = random.randint(0, n-1)
        sim = cosine_similarity(feat_list[idx1].reshape(1, -1), feat_list[idx2].reshape(1, -1))[0][0].item()
        random_vals.append(sim)
    return np.mean(consecutive_vals), np.mean(random_vals)


def custom_eval(make_agent, make_env, make_logger, args):
  assert args.from_checkpoint

  agent = make_agent()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  agg = elements.Agg()
  epstats = elements.Agg()
  episodes = defaultdict(elements.Agg)
  should_log = elements.when.Clock(args.log_every)
  policy_fps = elements.FPS()

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      isimage = (value.dtype == np.uint8) and (value.ndim == 3)
      if isimage and worker == 0:
        episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  episode_done = False
  def episode_tracker(tran, worker):
      global episode_done
      if tran['reward'] != 0:
         print(f"Reward at step {frame_idx}: {tran['reward']}")
      if tran['is_last']:
          episode_done = True

  def save_frame(tran, worker):
      global frame_idx

      img = tran['image']  # (64, 64, 3) uint8

      # Convert to PIL image
      im = Image.fromarray(img)

      # Save with sequential filename
      fname = os.path.join(outdir, f"frame_{frame_idx:06d}.png")
      im.save(fname)

      frame_idx += 1


  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=(not args.debug))
  driver.on_step(episode_tracker)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(logfn)
  driver.on_step(save_frame)
  

  cp = elements.Checkpoint()
  cp.agent = agent
  cp.load(args.from_checkpoint, keys=['agent'])

  embeds = []   # list of (B, D) arrays
  acts_log = [] # optional: store actions too

  def policy(carry, obs, **kwargs):
      carry, act, outs = agent.policy(carry, obs, mode='eval')
  
      deter = np.array(outs['dyn/deter'])  # (B, 8192)
      stoch = np.array(outs['dyn/stoch']).reshape(deter.shape[0], -1)  # (B, 32*64)
    #   emb = np.concatenate([deter, stoch], axis=-1).squeeze()  # (B, 10240)

      emb = deter.squeeze()
  
      embeds.append(emb)
      acts_log.append({k: np.array(v) for k, v in act.items()})
      return carry, act, outs

  print('Start evaluation')
#   policy = lambda *args: agent.policy(*args, mode='eval')
  driver.reset(agent.init_policy)
  while step < args.steps:
    driver(policy, steps=1)
    if should_log(step):
      logger.add(agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

  logger.close()
  print(len(embeds))
  compute_MDS(embeds, similarity_metric="cosine",
              title="DreamerV3 Embeddings Recurrent State", save_path=os.path.join("MDS_plots", "dreamerv3_mds_recurrent_state.png"))
  # print(embeds)
