//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// Physical constants
// Author: Arne Vansteenkiste.

import ()

// Physical constants
const (
	PI       = 3.14159265358979323846264338327950288
	Mu0      = 4.0 * PI * 1e-7    // Permeability of vacuum in J/Am2
	Epsilon0 = 8.854187817620E-12 // Permittivity of vacuum in C/Vm
	Gamma0   = 2.211E5            // Gyromagnetic ratio in m/As (actually γ*µ0)
	Kb       = 1.380650424E-23    // Boltzmann's constant in J/K
	MuB      = 9.2740091523E-24   // Bohr magneton in Am^2
	E        = 1.60217646E-19     // Electron charge in As
	H_bar    = 1.054571726E-34    // Reduced Planck's Constant in J*s
)
