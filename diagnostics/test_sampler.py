"""
test_sampler.py — verifies the noise properties claimed for CorrelatedSampler.

Regenerates the numbers quoted in mdn.py's docstring and in the paper:
    - lag-1 autocorrelation of the OU noise vs. white
    - sign-flip rate vs. the arccos(rho)/pi predicted for an AR(1) process
    - noise standard deviation (should sit at 1 by construction)
    - T=0 is bit-identical to sample_mdn's greedy path and consumes no RNG

Run:  python test_sampler.py
"""

import numpy as np
import tensorflow as tf

from osunator.mdn import (CorrelatedSampler, sample_mdn, split_mdn_params,
                          N_MIX, OUT_DIM, MDN_PARAMS)

N_DRAWS = 20000
RHO = 0.94
TEMPERATURE = 1.0


def flat_params():
    """One tick of MDN params with uniform weights, zero means, sigma from
    softplus(0). Zero means make the emitted sample equal sigma * T * noise,
    so the noise can be recovered by dividing out."""
    return np.zeros((1, 1, MDN_PARAMS), dtype=np.float32)


def summarize(name, seq, predicted_flip=None):
    seq = np.asarray(seq, dtype=float)
    lag1 = np.corrcoef(seq[:-1], seq[1:])[0, 1]
    flips = (np.sign(seq[1:]) != np.sign(seq[:-1])).mean()
    line = (f"{name:<24} lag-1 {lag1:+.4f}   flip rate {flips:.4f}"
            f"   std {seq.std():.4f}")
    if predicted_flip is not None:
        line += f"   (predicted flip {predicted_flip:.4f})"
    print(line)
    return lag1, flips, seq.std()


def main():
    params = flat_params()

    # sigma actually in force, for un-normalizing the emitted samples
    _, _, sigma = split_mdn_params(tf.convert_to_tensor(params))
    sigma_val = float(sigma.numpy().reshape(-1)[0])
    print(f"draws: {N_DRAWS}   rho: {RHO}   T: {TEMPERATURE}   sigma: {sigma_val:.4f}\n")

    # --- correlated sampler: read the OU state directly, and the samples ---
    s = CorrelatedSampler(temperature=TEMPERATURE, rho=RHO,
                          rng=np.random.default_rng(1))
    ou_noise, ou_samples = [], []
    for _ in range(N_DRAWS):
        out = s.sample(params)
        ou_noise.append(s.eps[0])                 # the noise state itself
        ou_samples.append(out[0, 0, 0])           # the emitted dx
    ou_samples = np.array(ou_samples) / (sigma_val * TEMPERATURE)

    # --- white baseline from sample_mdn, same shape of draw ---
    rng = np.random.default_rng(2)
    white_samples = np.array([
        sample_mdn(params, temperature=TEMPERATURE, rng=rng)[0, 0, 0]
        for _ in range(N_DRAWS)
    ]) / (sigma_val * TEMPERATURE)

    predicted = np.arccos(RHO) / np.pi
    print("noise properties")
    summarize("  OU state (eps)", ou_noise, predicted)
    summarize("  OU emitted samples", ou_samples, predicted)
    summarize("  white (sample_mdn)", white_samples)

    # --- T=0: greedy, deterministic, no RNG consumed ---
    print("\ndeterminism at T = 0")
    greedy_direct = sample_mdn(params, temperature=0.0)
    s0 = CorrelatedSampler(temperature=0.0, rng=np.random.default_rng(3))
    eps_before = s0.eps.copy()
    greedy_sampler = s0.sample(params)
    identical = np.array_equal(greedy_direct, greedy_sampler)
    state_untouched = np.array_equal(eps_before, s0.eps)
    print(f"  CorrelatedSampler matches sample_mdn greedy : {identical}")
    print(f"  OU state left unadvanced                    : {state_untouched}")

    repeat = CorrelatedSampler(temperature=0.0,
                               rng=np.random.default_rng(99)).sample(params)
    print(f"  independent of seed                         : "
          f"{np.array_equal(greedy_sampler, repeat)}")


if __name__ == "__main__":
    main()