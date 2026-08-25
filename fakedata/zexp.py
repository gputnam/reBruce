"""z-expansion axial form factor, ported from GENIE ZExpAxialFormFactorModel
(SBNSoftware/Generator @ v3_06_02_br).

    z(q2)  = (sqrt(Tcut - q2) - sqrt(Tcut - T0)) / (sqrt(Tcut - q2) + sqrt(Tcut - T0))
    FA(q2) = sum_{k=0}^{Kmax + (4 if Q4limit)} a_k z^k

q2 is GENIE's spacelike q2 = -Q2 < 0 (GeV^2). With Q4limit (all SBN tunes),
a_0 and a_{Kmax+1..Kmax+4} are solved from a_1..a_Kmax by FixQ4Limit() so that
FA(0) = gA (negative in GENIE) and FA falls as 1/Q^4 asymptotically. The
FixQ4Limit algebra below is a verbatim port.

Coefficient sets (SBN GENIE fork config/ZExpAxialFormFactorModel.xml and
config/AR23_20i/<tune>/CommonParam.xml QEL-FA0):
  - deuterium ("Default", Meyer et al. PRD 93 113015; the AR23_20i_00_000 CV)
  - minerva_nature (T. Cai et al., Nature 614 (2023) 48; tune AR23_20i_01_001)
  - lqcd (LQCD average; tune AR23_20i_02_000)
"""

import numpy as np

COEFF_SETS = {
    "deuterium": dict(
        a=[2.30, -0.6, -3.8, 2.3], t0=-0.28, tcut=0.1764, ga=-1.2670),
    "minerva_nature": dict(
        a=[1.50, -1.2, -0.1, 0.2], t0=-0.75, tcut=0.1764, ga=-1.2723),
    "lqcd": dict(
        a=[1.721, -0.31], t0=-0.5, tcut=0.161604, ga=-1.2754),
}


class ZExpAxialFF:
    def __init__(self, a, t0, tcut, ga, q4limit=True):
        """a: [a_1 .. a_Kmax]; ga = FA(0) (GENIE sign convention, negative)."""
        self.t0 = float(t0)
        self.tcut = float(tcut)
        self.ga = float(ga)
        self.kmax = len(a)
        self.q4limit = q4limit
        if q4limit:
            self.an = np.zeros(self.kmax + 5)
            self.an[1:self.kmax + 1] = a
            self._fix_q4limit()
        else:
            self.an = np.zeros(self.kmax + 1)
            self.an[1:] = a
            z0 = self.z(0.0)
            self.an[0] = ga - np.sum(self.an[1:] * z0 ** np.arange(1, self.kmax + 1))

    @classmethod
    def from_name(cls, name):
        return cls(**COEFF_SETS[name])

    def z(self, q2):
        """Expansion parameter; q2 = -Q2 <= 0 (GeV^2)."""
        num = np.sqrt(self.tcut - q2) - np.sqrt(self.tcut - self.t0)
        den = np.sqrt(self.tcut - q2) + np.sqrt(self.tcut - self.t0)
        return num / den

    def __call__(self, q2):
        """FA(q2) (GENIE sign convention: FA(0) = gA < 0)."""
        zp = self.z(q2)
        fa = np.zeros_like(np.asarray(zp, dtype=np.float64))
        for k in range(len(self.an)):
            fa = fa + self.an[k] * zp ** k
        return fa

    def _fix_q4limit(self):
        # verbatim port of ZExpAxialFormFactorModel::FixQ4Limit()
        an = self.an
        kmax = self.kmax
        kp4 = kmax + 4.0
        kp3 = kmax + 3.0
        kp2 = kmax + 2.0
        kp1 = kmax + 1.0
        kp0 = float(kmax)
        z0 = self.z(0.0)
        zkp4 = z0 ** int(kp4)
        zkp3 = z0 ** int(kp3)
        zkp2 = z0 ** int(kp2)
        zkp1 = z0 ** int(kp1)

        denom = (6.0 - kp4 * kp3 * kp2 * zkp1 + 3.0 * kp4 * kp3 * kp1 * zkp2
                 - 3.0 * kp4 * kp2 * kp1 * zkp3 + kp3 * kp2 * kp1 * zkp4)

        ks = np.arange(1, kmax + 1)
        b0 = np.sum(an[1:kmax + 1])
        b0z = -self.ga + np.sum(an[1:kmax + 1] * z0 ** ks)
        b1 = np.sum(ks * an[1:kmax + 1])
        b2 = np.sum(ks * (ks - 1) * an[1:kmax + 1])
        b3 = np.sum(ks * (ks - 1) * (ks - 2) * an[1:kmax + 1])

        an[int(kp4)] = (1.0 / denom) * (
            (b0 - b0z) * kp3 * kp2 * kp1
            + b3 * (-1.0 + 0.5 * kp3 * kp2 * zkp1 - kp3 * kp1 * zkp2
                    + 0.5 * kp2 * kp1 * zkp3)
            + b2 * (3.0 * kp1 - kp3 * kp2 * kp1 * zkp1
                    + kp3 * kp1 * (2 * kmax + 1) * zkp2 - kp2 * kp1 * kp0 * zkp3)
            + b1 * (-3.0 * kp2 * kp1 + 0.5 * kp3 * kp2 * kp2 * kp1 * zkp1
                    - kp3 * kp2 * kp1 * kp0 * zkp2
                    + 0.5 * kp2 * kp1 * kp1 * kp0 * zkp3))

        an[int(kp3)] = (1.0 / denom) * (
            -3.0 * (b0 - b0z) * kp4 * kp2 * kp1
            + b3 * (3.0 - kp4 * kp2 * zkp1 + (3.0 / 2.0) * kp4 * kp1 * zkp2
                    - 0.5 * kp2 * kp1 * zkp4)
            + b2 * (-3.0 * (3 * kmax + 4) + kp4 * kp2 * (2 * kmax + 3) * zkp1
                    - 3.0 * kp4 * kp1 * kp1 * zkp2 + kp2 * kp1 * kp0 * zkp4)
            + b1 * (3.0 * kp1 * (3 * kmax + 8) - kp4 * kp3 * kp2 * kp1 * zkp1
                    + (3.0 / 2.0) * kp4 * kp3 * kp1 * kp0 * zkp2
                    - 0.5 * kp2 * kp1 * kp1 * kp0 * zkp4))

        an[int(kp2)] = (1.0 / denom) * (
            3.0 * (b0 - b0z) * kp4 * kp3 * kp1
            + b3 * (-3.0 + 0.5 * kp4 * kp3 * zkp1 - (3.0 / 2.0) * kp4 * kp1 * zkp3
                    + kp3 * kp1 * zkp4)
            + b2 * (3.0 * (3 * kmax + 5) - kp4 * kp3 * kp2 * zkp1
                    + 3.0 * kp4 * kp1 * kp1 * zkp3
                    - kp3 * kp1 * (2 * kmax + 1) * zkp4)
            + b1 * (-3.0 * kp3 * (3 * kmax + 4) + 0.5 * kp4 * kp3 * kp3 * kp2 * zkp1
                    - (3.0 / 2.0) * kp4 * kp3 * kp1 * kp0 * zkp3
                    + kp3 * kp2 * kp1 * kp0 * zkp4))

        an[int(kp1)] = (1.0 / denom) * (
            -(b0 - b0z) * kp4 * kp3 * kp2
            + b3 * (1.0 - 0.5 * kp4 * kp3 * zkp2 + kp4 * kp2 * zkp3
                    - 0.5 * kp3 * kp2 * zkp4)
            + b2 * (-3.0 * kp2 + kp4 * kp3 * kp2 * zkp2
                    - kp4 * kp2 * (2 * kmax + 3) * zkp3 + kp3 * kp2 * kp1 * zkp4)
            + b1 * (3.0 * kp3 * kp2 - 0.5 * kp4 * kp3 * kp3 * kp2 * zkp2
                    + kp4 * kp3 * kp2 * kp1 * zkp3
                    - 0.5 * kp3 * kp2 * kp2 * kp1 * zkp4))

        an[0] = (1.0 / denom) * (
            -6.0 * b0z
            + b0 * (kp4 * kp3 * kp2 * zkp1 - 3.0 * kp4 * kp3 * kp1 * zkp2
                    + 3.0 * kp4 * kp2 * kp1 * zkp3 - kp3 * kp2 * kp1 * zkp4)
            + b3 * (-zkp1 + 3.0 * zkp2 - 3.0 * zkp3 + zkp4)
            + b2 * (3.0 * kp2 * zkp1 - 3.0 * (3 * kmax + 5) * zkp2
                    + 3.0 * (3 * kmax + 4) * zkp3 - 3.0 * kp1 * zkp4)
            + b1 * (-3.0 * kp3 * kp2 * zkp1 + 3.0 * kp3 * (3 * kmax + 4) * zkp2
                    - 3.0 * kp1 * (3 * kmax + 8) * zkp3 + 3.0 * kp2 * kp1 * zkp4))
