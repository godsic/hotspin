//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// modified by Mykola Dvornik

package engine

import (
	"fmt"
	. "hotspin-core/common"
	"strings"
)

const ArgDelim = ":"
// A physics module. Loading it adds various quantity nodes to the engine.
type Arguments struct {
	InsMap   map[string]string // the string-to-string map of the input quantities
	DepsMap map[string]string // the string-to-string map of the dependencies
	OutsMap  map[string]string // the string-to-string map of the output quantities
}

func (t *Arguments)Deps(name string)string{
	return GetVariable(t.DepsMap, name)
}

func (t *Arguments)Ins(name string)string{
	return GetVariable(t.InsMap, name)
}

func (t *Arguments)Outs(name string)string{
	return GetVariable(t.OutsMap, name)
}

func GetVariable(argMap map[string]string, name string) string {
	key := argMap[name]
	if len(key) == 0 {
		panic(InputErr("The requested variable: " + name + " is not avalible"))
	}
	return key
}

type Module struct {
	Name         string                           	// Name to identify to module to the machine
	Description  string                           	// Human-readable description of what the module does
	Args         Arguments                        	// The map of arguments and their default values
	LoadFunc     func(e *Engine, args ...Arguments)   // Loads this module's quantities and dependencies into the engine
}

// Map with registered modules
var modules map[string]Module = make(map[string]Module)

// Registers a module in the list of known modules.
// Each module should register itself in its init() function.
func RegisterModule(name, description string, loadfunc func(e *Engine)) {
	if _, ok := modules[name]; ok {
		panic(InputErr("module " + name + " already registered"))
	}
	
	loadfuncargs := func(e *Engine, args ...Arguments) {
		loadfunc(e)
	}
	
	modules[name] = Module{name, description, Arguments{}, loadfuncargs}
}

func RegisterModuleArgs(name, description string, args Arguments, loadfuncargs func(e *Engine, args ...Arguments)) {
	if _, ok := modules[name]; ok {
		panic(InputErr("module " + name + " already registered"))
	}
	modules[name] = Module{name, description, args, loadfuncargs}
}

func GetModule(name string) Module {
	module, ok := modules[name]

	if !ok {
		opts := []string{}
		for k, _ := range modules {
			opts = append(opts, k)
		}
		panic(InputErr(fmt.Sprint("Unknown module:", name, " Options: ", opts)))
	}
	return module
}

func ParseArgument(m map[string]string, v string) {
	pair := strings.Split(v, ArgDelim)
	if len(pair) > 2 {
		panic(InputErr("Cannot parse user-defined variable: " + v ))
	}
	
	_, ok := m[pair[0]]
	if !ok {
		panic(InputErr("Cannot assign non-existing variable: " + v ))
	}
	
	m[pair[0]] = pair[1]
}


func GetParsedArgumentsMap(module Module, in,deps,out []string) Arguments {
	arg := module.Args
	
	for _,val := range in {
		ParseArgument(arg.InsMap, val)
	}
	for _,val := range deps {
		ParseArgument(arg.DepsMap, val)
	}
	for _,val := range out {
		ParseArgument(arg.OutsMap, val)
	}
	
	return arg
}
