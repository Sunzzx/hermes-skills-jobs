"""Unit tests for sph-hydrodynamics: 2D Smoothed Particle Hydrodynamics engine."""

import math
import os
from importlib.machinery import SourceFileLoader

bin_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../bin/sph-hydrodynamics")
)
eng = SourceFileLoader("sph_hydrodynamics", bin_path).load_module()


def test_kernels_registered():
    assert "cubic_spline" in eng.KERNELS
    assert "wendland_c2" in eng.KERNELS


def test_cubic_spline_partition_of_unity():
    pu = eng.kernel_partition_of_unity(eng.KERNELS["cubic_spline"], h=0.05, n=25)
    assert 0.9 <= pu <= 1.1


def test_wendland_partition_of_unity():
    pu = eng.kernel_partition_of_unity(eng.KERNELS["wendland_c2"], h=0.05, n=25)
    assert 0.9 <= pu <= 1.1


def test_kernel_gradient_is_odd():
    for kern in eng.KERNELS.values():
        assert eng.kernel_gradient_odd(kern, h=0.04) < 1e-12


def test_kernel_compact_support():
    h = 0.05
    for kern in eng.KERNELS.values():
        assert kern.W(2.0 * h + 1e-6, h) == 0.0
        assert kern.W(0.0, h) > 0.0


def test_tait_eos_rest_pressure_zero():
    cfg = eng.SPHConfig(rho0=1000.0, c0=10.0, gamma=7.0, eos="tait")
    w = eng.SPHWorld(cfg, [])
    assert abs(w.pressure(1000.0)) < 1e-9
    assert w.pressure(1100.0) > w.pressure(1000.0)


def test_dam_break_mass_conservation():
    out = eng.run_simulation(scenario="dam_break", steps=20, seed=1)
    assert out["n_fluid"] > 50
    assert out["mass_rel_err"] < 1e-12
    assert math.isfinite(out["e_tot_1"])


def test_hydrostatic_stable_density():
    out = eng.run_simulation(scenario="hydrostatic", steps=20, kernel="wendland_c2")
    assert out["mass_rel_err"] < 1e-12
    # weakly compressible: density stays O(rho0)
    assert out["rho_max"] < 3.0 * 1000.0
    assert out["rho_mean"] > 0.5 * 1000.0


def test_droplet_finite_and_bounded():
    out = eng.run_simulation(scenario="droplet", steps=15)
    assert math.isfinite(out["v_max"])
    assert out["v_max"] < 50.0
    assert out["mass_rel_err"] < 1e-12


def test_shear_periodic_mass():
    out = eng.run_simulation(scenario="shear", steps=12)
    assert out["mass_rel_err"] < 1e-12
    assert out["n_fluid"] > 50


def test_spatial_hash_finds_self():
    cfg, parts = eng.scenario_hydrostatic(n=40)
    world = eng.SPHWorld(cfg, parts)
    world._rebuild_hash()
    assert 0 in world._hash.neighbours(parts[0].x, parts[0].y)


def test_build_world_and_step():
    world = eng.build_world("dam_break")
    d0 = world.diagnostics(world.suggest_dt())
    d1 = world.step()
    assert d1.step == 1
    assert d1.t > 0
    assert abs(d1.mass - d0.mass) < 1e-9


def test_ascii_field_nonempty():
    world = eng.build_world("droplet")
    art = world.ascii_field(nx=20, ny=10)
    assert isinstance(art, str)
    assert len(art.splitlines()) == 10


def test_compare_kernels_schema():
    out = eng.compare_kernels(scenario="hydrostatic", steps=10)
    assert len(out["kernels"]) == 2
    names = {r["kernel"] for r in out["kernels"]}
    assert names == {"cubic_spline", "wendland_c2"}


def test_theory_ledger():
    th = eng.theory_ledger(8000)
    assert th["n_particles"] == 8000
    assert th["memory_mib"] > 0
    assert th["pair_flops_per_step_est"] > 0


def test_selftest_exit_zero():
    assert eng.selftest() == 0


def test_suggest_dt_positive():
    world = eng.build_world("shear")
    dt = world.suggest_dt()
    assert dt > 0
    assert dt < 1.0


def test_scenarios_registry():
    assert set(eng.SCENARIOS) >= {"dam_break", "hydrostatic", "droplet", "shear"}
