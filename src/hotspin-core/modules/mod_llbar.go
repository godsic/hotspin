//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

import (
	. "hotspin-core/engine"
)

// Register this module

func init() {
	RegisterModule("llbar", "Landau-Lifshitz-Baryakhtar equation", LoadLLBar)
}

// The torque quant contains the Landau-Lifshitz-Baryakhtar torque τ acting on the reduced magnetization m = M/Msat0T0, where Msat0T0 is the zero-temperature value of saturation magnetization
//	d m / d t =  τ
// with unit
//	[τ] = 1/s
// Thus:
//	τ = gammaLL[ ( \lambda\_ij H + M x M x \mu\_ij H - \lambdae\_e laplacian(H) ]
// To keep numbers from getting extremely large or small,
// the multiplier is set to gamma, so the array stores τ/gamma

func LoadLLBar(e *Engine) {

	LoadFullMagnetization(e)

	llbar_RHS := e.AddNewQuant("llbar_RHS", VECTOR, FIELD, Unit("/s"), "The Right Hand Side of Landau-Lifshitz-Baryakhtar equation")
	llbar_RHS.SetUpdater(NewSumUpdater(llbar_RHS))

	e.AddPDE1("m", "llbar_RHS")

}
