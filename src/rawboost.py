"""
RawBoost-style data augmentation, per Section 6.2 of the paper:

  "RawBoost adds variation in the form of: i) linear and non-linear
  convolutive noise; ii) impulsive signal-dependent additive noise;
  iii) stationary signal-independent additive noise."

  "For the LA database... a combination of linear and non-linear
  convolutive noise and impulsive signal-dependent additive noise
  strategies work best."

  "for the DF database, DA works best using stationary signal-independent
  additive, randomly coloured noise."

ENGINEERING NOTE: the exact parameter distributions live in the separate
RawBoost paper/repo (TakHemlata/RawBoost-antispoofing), which this paper
cites but doesn't inline. The implementations below follow the three
described noise *families* with reasonable parameter ranges. Swap in the
official repo's exact ranges if you need byte-exact DA numbers.
"""
import numpy as np


def _normalize(x, eps=1e-9):
    return x / (np.max(np.abs(x)) + eps)


def linear_nonlinear_convolutive_noise(x, sr=16000, n_bands=5, nonlinear_order=5,
                                        seed=None):
    """Simulates convolutive channel/device distortion: a random FIR filter
    (linear) followed by a mild nonlinear (harmonic) distortion term, both of
    which are the kind of variation telephony/codec transmission introduces.
    """
    rng = np.random.RandomState(seed)
    # Linear part: random minimum-phase-ish FIR from a handful of random band gains
    n_taps = 21
    band_gains = rng.uniform(0.5, 1.5, size=n_bands)
    freqs = np.linspace(0, 1, n_bands)
    taps = np.interp(np.linspace(0, 1, n_taps), freqs, band_gains)
    taps = taps / np.sum(np.abs(taps))
    y = np.convolve(x, taps, mode="same")

    # Non-linear part: odd-order polynomial distortion (harmonic generation)
    y_nl = np.copy(y)
    coeff = rng.uniform(0.05, 0.2)
    for order in range(3, nonlinear_order + 1, 2):
        y_nl = y_nl + coeff * (y ** order) / order

    return _normalize(y_nl).astype(np.float32)


def impulsive_signal_dependent_noise(x, seed=None, density=0.003, gain_range=(2.0, 6.0)):
    """Sparse, signal-amplitude-dependent impulsive noise — models transient
    channel artefacts (clicks/pops) whose magnitude scales with local signal
    energy, matching the 'signal-dependent' description in the paper.
    """
    rng = np.random.RandomState(seed)
    y = np.copy(x)
    n = len(x)
    n_impulses = max(1, int(n * density))
    idx = rng.choice(n, size=n_impulses, replace=False)
    local_energy = np.abs(x[idx]) + 1e-4
    gains = rng.uniform(*gain_range, size=n_impulses)
    signs = rng.choice([-1, 1], size=n_impulses)
    y[idx] = y[idx] + signs * gains * local_energy
    return _normalize(y).astype(np.float32)


def stationary_coloured_additive_noise(x, seed=None, snr_db_range=(10, 25), color_alpha=1.0):
    """Stationary, signal-INdependent additive noise with a 1/f^alpha coloured
    spectrum — models the smoother, compression-artefact-like noise the
    paper says works best for the DF (compression-heavy) database.
    """
    rng = np.random.RandomState(seed)
    n = len(x)
    white = rng.randn(n)
    # Colour it: shape the noise spectrum as 1/f^alpha
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if n > 1 else 1.0  # avoid div-by-zero at DC
    shaping = 1.0 / (freqs ** (color_alpha / 2.0))
    shaping = shaping / np.max(shaping)
    coloured = np.fft.irfft(spec * shaping, n=n)
    coloured = _normalize(coloured)

    sig_power = np.mean(x ** 2) + 1e-9
    snr_db = rng.uniform(*snr_db_range)
    noise_power = sig_power / (10 ** (snr_db / 10))
    coloured = coloured * np.sqrt(noise_power / (np.mean(coloured ** 2) + 1e-9))

    y = x + coloured
    return _normalize(y).astype(np.float32)


def rawboost_la(x, sr=16000, seed=None):
    """LA-optimised config: linear+non-linear convolutive noise, then
    impulsive signal-dependent additive noise (Section 6.2)."""
    rng = np.random.RandomState(seed)
    x = linear_nonlinear_convolutive_noise(x, sr=sr, seed=rng.randint(1 << 30))
    x = impulsive_signal_dependent_noise(x, seed=rng.randint(1 << 30))
    return x


def rawboost_df(x, sr=16000, seed=None):
    """DF-optimised config: stationary signal-independent additive coloured
    noise (Section 6.2)."""
    return stationary_coloured_additive_noise(x, seed=seed)


RAWBOOST_CONFIGS = {
    "la": rawboost_la,
    "df": rawboost_df,
    "none": lambda x, sr=16000, seed=None: x,
}
