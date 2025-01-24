"""
Shows an example of loading SIMSOPT output trajectory 
and classifying it as passing particle.
"""
import numpy as np
import matplotlib.pyplot as plt
from simsopt.field import BoozerRadialInterpolant
from trajectory_tools import (
    Trajectory,
    compute_locally_trapped_condition,
    classify_trajectory_from_locally_trapped_sections,
    get_max_modB_spline,
    TrajectoryType
)
import booz_xform as bx

equil = bx.Booz_xform()
equil.verbose = False
equil.read_boozmn('boozmn_qh_50.nc')

bri = BoozerRadialInterpolant(
        equil=equil,
        order=3,
        mpol=equil.mboz,
        ntor=equil.nboz,
        N=-4,
        enforce_vacuum=False,
        rescale =False,
        no_K=False
)

tj = Trajectory('example_trapped_trajectory_tys.txt')
tj.set_field_helicity(-4)

trapped_sections, passing_sections = compute_locally_trapped_condition(
    equilibrium_field=bri,
    trajectories=tj
)[0]

particle_type = classify_trajectory_from_locally_trapped_sections(
    trapped_sections, passing_sections
)
print(f'{particle_type=}')
assert particle_type == TrajectoryType.TRAPPED

fig1, ax1 = plt.subplots(figsize=(6, 5), frameon=False)
fig2, ax2 = plt.subplots(figsize=(6, 5), frameon=False)

ax1.plot(
    np.sqrt(tj.s) * np.cos(tj.chi),
    np.sqrt(tj.s) * np.sin(tj.chi),
    label='Passing',
    color = 'g'
)
ax2.plot(tj.t, tj.s, color='g')

ax1.set_xlabel('$\sqrt{s} \cos{\chi}$')
ax1.set_ylabel('$\sqrt{s} \sin{\chi}$')
ax1.set_xlim([-1, 1])
ax1.set_ylim([-1, 1])

ax2.set_xlabel('$t$')
ax2.set_ylabel('$s$')
ax2.set_xlim([tj.t[0], tj.t[-1]])
ax2.set_ylim([-1, 1])

plt.sca(ax1)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(f'pseudo_cartesian_plot_for_trapped_particle_example.png', dpi=150)
plt.close()

plt.sca(ax2)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(f'timeplot_for_trapped_particle_example.png', dpi=150)
