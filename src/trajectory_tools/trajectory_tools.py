from typing import List, Tuple, Union
from enum import Enum
import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import booz_xform as bx
from simsopt.field.boozermagneticfield import (
    BoozerRadialInterpolant
)

from simsopt.util.constants import (
    ALPHA_PARTICLE_MASS as MASS, 
    FUSION_ALPHA_PARTICLE_ENERGY as KINETIC_ENERGY,
    ALPHA_PARTICLE_CHARGE as CHARGE
)

class TrajectoryType(Enum):
    PASSING = "PASSING"
    TRAPPED = "TRAPPED"
    MIXED = "MIXED"

class Trajectory:
    def __init__(self, filename):
        pt = np.loadtxt(
                filename,
                dtype = [
                    ('time','f8'),
                    ('s', 'f8'),
                    ('theta', 'f8'),
                    ('zeta', 'f8'),
                    ('vp', 'f8'),
                    ('t', 'f8')
                    ]
                )
        self.filename = filename
        self.t = pt['t']
        self.s = pt['s']
        self.th = pt['theta']
        self.zt = pt['zeta']
        self.vp = pt['vp']
        self.chi = None
        self.field_helicity = None

    def set_field_helicity(self, field_helicity):
        self.field_helicity = field_helicity
        self.chi = self.th - self.field_helicity * self.zt

    def is_passing(self):
        assert len(self.vp) > 1, "Too few vp entries to check sign"
        for vp in self.vp:
            if vp[0] * vp[0] < 0:
                return False
        return True

    def __repr__(self):
        return f'({self.filename=}' + f'; t_end={self.t[-1]})'
        
def get_max_modB_spline(
field : BoozerRadialInterpolant
) -> CubicSpline:
    '''
    Computes maximum of B on each flux surface, and
    returns the spline interpolation for that
    '''
    theta_grid = np.linspace(0, 2*np.pi, 5*field.mpol, endpoint=False)
    zeta_grid = np.linspace(0, 2*np.pi, 5*field.ntor, endpoint=False)
    theta_grid, zeta_grid = np.meshgrid(theta_grid, zeta_grid)
    theta_flat = theta_grid.flatten()
    zeta_flat = zeta_grid.flatten()
    max_modB = []
    s = np.linspace(0,1,200)
    print('Building interpolant for flux-surface B maximum...')
    for s_value in s:
        points = np.zeros((len(theta_flat), 3))
        points[:, 0] = s_value
        points[:, 1] = theta_flat
        points[:, 2] = zeta_flat
        field.set_points(points)
        max_modB.append(np.max(field.modB()))
        print(f'\t s = {s_value:.2f}, max(|B|) = {max_modB[-1]:.2e}', end='\r')
    print('\033[K', end='')
    print('\033[FBuilding interpolant for', 'flux-surface B maximum: Done')
    return CubicSpline(s, max_modB)

def identify_island_boundaries(
    trapped: np.ndarray
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Identify shore indices for trapped and passing sections.

    Args:
        is_trapped (np.ndarray): A boolean array indicating whether a point is trapped or not.

    Returns:
        Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]: A tuple containing two lists of tuples representing the shore indices
            for trapped and passing sections respectively.
    """
    left_shore = 0
    passing_shores = []
    trapped_shores = []
    for i, is_trapped in enumerate(trapped[1:], start=1):
        if trapped[left_shore] != is_trapped:
            right_shore = i
            if trapped[left_shore]:
                trapped_shores.append((left_shore, right_shore))
            else:
                passing_shores.append((left_shore, right_shore))
            left_shore = i
    if trapped[left_shore]:
        trapped_shores.append((left_shore, len(trapped)))
    else:
        passing_shores.append((left_shore, len(trapped)))
    return trapped_shores, passing_shores

def compute_locally_trapped_condition(
    equilibrium_field : BoozerRadialInterpolant,
    trajectories : Union[List[Trajectory], Trajectory]
) -> List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:  
    '''
    For each point alont the Trajectory, computes whether it is locally
    trapped or not.
    
    For each trajectory, returns two lists of trajectory ends 
    – index pairs for the firts and last trajectory points – 
    for localy trapped and passing sections.
    '''
    if isinstance(trajectories, Trajectory):
        trajectories = [trajectories]
    elif (
        isinstance(trajectories, list) 
        and 
        all(isinstance(item, Trajectory)for item in trajectories)
    ):
        pass
    else:
        raise ValueError("Expected a Trajectory or a List[Trajectory], got something else")
    max_modB_spline = get_max_modB_spline(equilibrium_field)
    locally_trapped_and_passing_island_lists = []
    for tj in trajectories:
        points = np.zeros((len(tj.s), 3))
        points[:, 0] = tj.s
        points[:, 1] = tj.th
        points[:, 2] = tj.zt
        equilibrium_field.set_points(points)
        modB = equilibrium_field.modB()[0][0]
        mu = (KINETIC_ENERGY - MASS*tj.vp[0]*tj.vp[0]/2)/modB
        mu_set = True
        trapped = KINETIC_ENERGY <= mu*max_modB_spline(tj.s)
        trapped_shores, passing_shores = identify_island_boundaries(trapped)
        locally_trapped_and_passing_island_lists.append(
            (trapped_shores, passing_shores)
        )
    return locally_trapped_and_passing_island_lists

def classify_trajectory_from_locally_trapped_sections(
    trapped_shores : List[Tuple[int, int]],
    passing_shores : List[Tuple[int, int]]
) -> TrajectoryType:
    if not trapped_shores:
        return TrajectoryType.PASSING
    elif not passing_shores:
        return TrajectoryType.TRAPPED
    else:
        return TrajectoryType.MIXED

class TestIdentifyIslandBoundaries(unittest.TestCase):
    def test_single_true(self):
        trapped = np.array([True])
        trapped_shores, passing_shores = identify_island_boundaries(trapped)
        self.assertEqual(trapped_shores, [(0, 1)])
        self.assertEqual(passing_shores, [])
    
    def test_double_true(self):
        trapped = np.array([True, True])
        trapped_shores, passing_shores = identify_island_boundaries(trapped)
        self.assertEqual(trapped_shores, [(0, 2)])
        self.assertEqual(passing_shores, [])
    
    def test_false_true_false(self):
        trapped = np.array([False, True, False])
        trapped_shores, passing_shores = identify_island_boundaries(trapped)
        self.assertEqual(trapped_shores, [(1, 2)])
        self.assertEqual(passing_shores, [(0, 1), (2, 3)])
    
    def test_true_true_false(self):
        trapped = np.array([True, True, False])
        trapped_shores, passing_shores = identify_island_boundaries(trapped)
        self.assertEqual(trapped_shores, [(0, 2)])
        self.assertEqual(passing_shores, [(2, 3)])
    
    def test_single_false(self):
        trapped = np.array([False])
        trapped_shores, passing_shores = identify_island_boundaries(trapped)
        self.assertEqual(trapped_shores, [])
        self.assertEqual(passing_shores, [(0, 1)])
    
if __name__ == '__main__':
    unittest.main()
