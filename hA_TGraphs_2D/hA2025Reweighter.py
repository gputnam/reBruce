import ROOT
import math


class hA2025Reweighter:
    """


    Inputs:
        KE   : pion kinetic energy in MeV
        fate : "cex", "abs", "inel", or "pipro"
        A    : target/remnant mass number

    Weight:
        FracADep_2025 / FracADep_2018
    """
    
    def __init__(self, file2018, file2025):

        self.f2018 = ROOT.TFile.Open(file2018, "READ")
        self.f2025 = ROOT.TFile.Open(file2025, "READ")

        if not self.f2018 or self.f2018.IsZombie():
            raise RuntimeError(f"Could not open {file2018}")

        if not self.f2025 or self.f2025.IsZombie():
            raise RuntimeError(f"Could not open {file2025}")

        # ------------------------------------------------------------
        # hA2018:
        # These graphs already contain fate fractions.
        # ------------------------------------------------------------
        self.ha18 = {
            "cex":   self.f2018.Get("TfracPipA_CEx"),
            "inel":  self.f2018.Get("TfracPipA_Inelas"),
            "abs":   self.f2018.Get("TfracPipA_Abs"),
            "pipro": self.f2018.Get("TfracPipA_PiPro"),
        }

        # ------------------------------------------------------------
        # hA2025:
        # These graphs contain log(cross section).
        # ------------------------------------------------------------
        self.ha25 = {
            "cex":   self.f2025.Get("TPipA_CEx"),
            "inel":  self.f2025.Get("TPipA_Inelas"),
            "abs":   self.f2025.Get("TPipA_Abs"),
            "pipro": self.f2025.Get("TPipA_PiPro"),
            "tot":   self.f2025.Get("TPipA_Tot"),
        }

        # Check that everything was loaded
        for model, graphs in [("hA2018", self.ha18),
                              ("hA2025", self.ha25)]:
            for name, graph in graphs.items():
                if not graph:
                    raise RuntimeError(
                        f"Could not find {name} graph for {model}"
                    )

    @staticmethod
    def _clamp(KE, A):
        """
        Match GENIE FracADep boundary handling:
            1 <= KE <= 999 MeV
            A <= 208
        """
        KE = max(1.0, float(KE))
        KE = min(999.0, KE)

        A = min(208.0, float(A))

        return KE, A

    def frac2018(self, KE, A):
        """
        

        Returns dictionary:
            {
                "cex": ...,
                "inel": ...,
                "abs": ...,
                "pipro": ...
            }
        """

        KE, A = self._clamp(KE, A)

        # TGraph2D convention in GENIE:
        # Interpolate(target A, kinetic energy)
        raw = {
            fate: graph.Interpolate(A, KE)
            for fate, graph in self.ha18.items()
        }

        # GENIE renormalizes the fractions
        total = sum(raw.values())

        if total == 0.0:
            return {
                "cex": 0.0,
                "inel": 0.0,
                "abs": 0.0,
                "pipro": 0.0,
            }

        return {
            fate: value / total
            for fate, value in raw.items()
        }

    def frac2025(self, KE, A):
      
        KE, A = self._clamp(KE, A)

        # Graphs contain log(cross section)
        log_xs = {
            fate: graph.Interpolate(A, KE)
            for fate, graph in self.ha25.items()
        }

        # GENIE uses exp(log_xs)
        xs = {
            fate: math.exp(value)
            for fate, value in log_xs.items()
        }

        # First construct component / total fractions
        tot_xs = xs["tot"]

        if tot_xs == 0.0:
            return {
                "cex": 0.0,
                "inel": 0.0,
                "abs": 0.0,
                "pipro": 0.0,
            }

        raw_frac = {
            "cex":   xs["cex"]   / tot_xs,
            "inel":  xs["inel"]  / tot_xs,
            "abs":   xs["abs"]   / tot_xs,
            "pipro": xs["pipro"] / tot_xs,
        }

        # GENIE renormalizes again
        total = sum(raw_frac.values())

        if total == 0.0:
            return {
                "cex": 0.0,
                "inel": 0.0,
                "abs": 0.0,
                "pipro": 0.0,
            }

        return {
            fate: value / total
            for fate, value in raw_frac.items()
        }

    def weight(self, KE, fate, A=40):
        """
        Return hA2025 / hA2018 fate weight.
        """

        fate = fate.lower()

        aliases = {
            "cex": "cex",
            "charge_exchange": "cex",
            "chargeexchange": "cex",

            "abs": "abs",
            "absorption": "abs",

            "inel": "inel",
            "inelas": "inel",
            "inelastic": "inel",

            "pipro": "pipro",
            "piprod": "pipro",
            "pion_production": "pipro",
        }

        if fate not in aliases:
            raise ValueError(
                f"Unknown fate '{fate}'. "
                "Use cex, abs, inel, or pipro."
            )

        fate = aliases[fate]

        f18 = self.frac2018(KE, A)[fate]
        f25 = self.frac2025(KE, A)[fate]

        # Same treatment as GReWeighthA2025
        if f18 != 0.0:
            weight = f25 / f18
        else:
            weight = 1.0

        return weight

    def print_info(self, KE, fate, A=40):
        """
        Useful for debugging / comparing against GENIE output.
        """

        fate = fate.lower()

        f18_all = self.frac2018(KE, A)
        f25_all = self.frac2025(KE, A)

        aliases = {
            "cex": "cex",
            "abs": "abs",
            "inel": "inel",
            "inelas": "inel",
            "pipro": "pipro",
            "piprod": "pipro",
        }

        fate = aliases[fate]

        f18 = f18_all[fate]
        f25 = f25_all[fate]

        w = 1.0 if f18 == 0 else f25 / f18

        print(f"KE      = {KE:.3f} MeV")
        print(f"A       = {A}")
        print(f"fate    = {fate}")
        print()
        print(f"hA2018 fraction = {f18:.8f}")
        print(f"hA2025 fraction = {f25:.8f}")
        print(f"weight          = {w:.8f}")

        return w

    def debug2018(self, KE, A):

        KE, A = self._clamp(KE, A)

        print(f"ROOT version = {ROOT.gROOT.GetVersion()}")
        print(f"KE used = {KE}")
        print(f"A used  = {A}")
        print()

        raw = {}

        for fate, graph in self.ha18.items():
            value = graph.Interpolate(A, KE)
            raw[fate] = value

            print(
                f"{fate:6s} : "
                f"Interpolate({A}, {KE}) = {value:.12f}"
            )

        total = sum(raw.values())

        print()
        print(f"sum = {total:.12f}")

        print("\nNormalized:")

        for fate, value in raw.items():
            print(
                f"{fate:6s} : {value / total:.12f}"
            )