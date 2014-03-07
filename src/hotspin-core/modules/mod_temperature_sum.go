//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

import (
	. "hotspin-core/engine"
	"hotspin-core/gpu"
)

const TSumName = "Teff"
const RCorrName = "R"

// Register this module
func init() {
	RegisterModule("temperature/sum", "Weighted correlated sum of baths temperatures.", LoadTempSum)
}

func LoadTempSum(e *Engine) {
	LoadTemp(e, EtempName)
	LoadTemp(e, LtempName)
	T := e.AddNewQuant(TSumName, SCALAR, FIELD, Unit("K"), "Weighted correlated sum of baths temperatures")
	R := e.AddNewQuant(RCorrName, SCALAR, FIELD, Unit(""), "Correlation coefficient")
	Te := e.Quant(EtempName)
	Tl := e.Quant(LtempName)
	lambda := e.Quant("λ∥")
	mu := e.Quant("μ∥")
	T.SetUpdater(&TSumUpdater{T: T, Te: Te, Tl: Tl, lambda: lambda, mu: mu, R: R})
	e.Depends(TSumName, EtempName, LtempName, "λ∥", "μ∥", RCorrName)
}

type TSumUpdater struct {
	T, Te, Tl, lambda, mu, R *Quant
}

func (u *TSumUpdater) Update() {
	gpu.WeightedAverage(u.T.Array(),
		u.Te.Array(), u.Tl.Array(),
		u.lambda.Array(), u.mu.Array(),
		u.R.Array(),
		u.lambda.Multiplier()[0], u.mu.Multiplier()[0],
		u.R.Multiplier()[0])
}
