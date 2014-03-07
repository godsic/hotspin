//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// This file implements the micromagnetism meta-module
// Author: Arne Vansteenkiste

import (
	"fmt"
	. "hotspin-core/common"
	. "hotspin-core/engine"
)

var inMaxTorque = map[string]string{
	"": "",
}

var depsMaxTorque = map[string]string{
	"torque":      "torque",
}

var outMaxTorque = map[string]string{
	"maxtorque": "maxtorque",
}

// Register this module
func init() {
	args := Arguments{inMaxTorque, depsMaxTorque, outMaxTorque}
	RegisterModuleArgs("maxtorque", "Calculates maximum torque for the given time", args, LoadMaxTorqueArgs)
}

func LoadMaxTorqueArgs(e *Engine, args ...Arguments) {
	
	// make it automatic !!!
	var arg Arguments
	
	if len(args) == 0 {
		arg = Arguments{inMaxTorque, depsMaxTorque, outMaxTorque}
	} else {
		arg = args[0]
	}
	//
	
	if e.HasQuant(arg.Deps("torque")) {
		torque := e.Quant(arg.Deps("torque"))
		maxtorque := e.AddNewQuant(arg.Outs("maxtorque"), SCALAR, VALUE, torque.Unit(), "Maximum |torque|")
		e.Depends(arg.Outs("maxtorque"), arg.Deps("torque"))
		maxtorque.SetUpdater(NewMaxNormUpdater(torque, maxtorque))
	} else {
		panic(InputErr(fmt.Sprint("maxtorque module should be loaded after micromagnetic equation module")))
	}
}
