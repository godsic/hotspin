//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements the sum of Quantities.
// Author: Arne Vansteenkiste

import (
	. "hotspin-core/common"
)

type AddTermUpdater struct {
	SumUpdater
	origUpdater Updater
}

func NewAddTermUpdater(orig *Quant) *AddTermUpdater {
	return &AddTermUpdater{SumUpdater{orig, nil, nil}, orig.Updater()}
}

func AddTermToQuant(sumQuant, term *Quant) {
	sumUpd, ok := sumQuant.GetUpdater().(SumNode)
	if !ok {
		upd := NewAddTermUpdater(sumQuant)
		sumUpd = upd
		sumQuant.updater = upd
	}
	sumUpd.AddParent(term.Name())
	Log("Adding quantity", term.FullName(), "to", sumQuant.Name())
}

func (u *AddTermUpdater) Update() {
	u.origUpdater.Update()
	u.addTerms()
}
