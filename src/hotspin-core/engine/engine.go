//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Arne Vansteenkiste

import (
	"fmt"
	. "hotspin-core/common"
	"hotspin-core/gpu"
	"path"
	"strings"
)

// The global simulation engine
var engine Engine

// Returns the global simulation engine
func GetEngine() *Engine {
	return &engine
}

// Engine is the heart of a multiphysics simulation.
// The engine stores named quantities like "m", "B", "alpha", ...
// An acyclic graph structure consisting of interconnected quantities
// determines what should be calculated and when.
type Engine struct {
	size3D_        [3]int            // INTENRAL
	size3D         []int             // size of the FD grid, nil means not yet set
	cellSize_      [3]float64        // INTENRAL
	cellSize       []float64         // size of the FD cells, nil means not yet set
	periodic_      [3]int            // INTERNAL
	periodic       []int             // periodicity in each dimension
	set_periodic_  bool              // INTERNAL: periodic already set?
	quantity       map[string]*Quant // maps quantity names onto their data structures
	equation       []Equation        // differential equations connecting quantities
	solver         Solver            // the solver simultaneously steps all equations forward in time
	time           *Quant            // time quantity is always present
	dt             *Quant            // time step quantity is always present
	step           *Quant            // number of time steps been taken
	timer          Timer             // For benchmarking
	modules        []Module          // loaded modules
	crontabs       map[int]Notifier  // periodical jobs, indexed by handle
	outputTables   map[string]*Table // open output table files, indexed by file name
	_outputID      int               // index for output numbering
	_lastOutputT   float64           // time of last output ID increment
	_handleCount   int               // used to generate unique handle IDs for various object passed out
	outputDir      string            // output directory
	filenameFormat string            // Printf format string for file name numbering. Must consume one integer.
}

// Initializes the global simulation engine
func Init() {
	(&engine).init()
}

// initialize
func (e *Engine) init() {
	e.periodic = e.periodic_[:]

	e.quantity = make(map[string]*Quant)

	// special quantities time and dt are always present
	e.AddNewQuant("t", SCALAR, VALUE, Unit("s"))
	e.time = e.Quant("t")

	e.AddNewQuant("dt", SCALAR, VALUE, Unit("s"))
	e.dt = e.Quant("dt")
	e.dt.SetVerifier(Positive)

	e.AddNewQuant("step", SCALAR, VALUE, Unit(""))
	e.step = e.Quant("step")

	e.equation = make([]Equation, 0, 1)
	e.solver = nil
	e.modules = make([]Module, 0)
	e.crontabs = make(map[int]Notifier)
	e.outputTables = make(map[string]*Table)
	e.filenameFormat = "%06d"
	e.timer.Start()
}

// Shuts down the engine. Closes all open files, etc.
func (e *Engine) Close() {
	for _, t := range e.outputTables {
		t.Close()
	}
}

//__________________________________________________________________ I/O

// Gets an ID number to identify the current time. Used to number output files. E.g. the 7 in "m000007.omf". Files with the same OutputID correspond to the same simulation time.
func (e *Engine) OutputID() int {
	t := e.time.Scalar()
	if t != e._lastOutputT {
		e._lastOutputT = t
		e._outputID++
	}
	return e._outputID
}

// Returns ++_handleCount. Used to identify objects like crontabs so they can later by manipulated through this ID.
func (e *Engine) NewHandle() int {
	e._handleCount++ // Let's not use 0 as a valid handle.
	return e._handleCount
}

// Resolves a file name relative to the output directory,
// unless the name begins with a /.
func (e *Engine) Relative(filename string) string {
	if filename[0] == '/' || strings.HasPrefix(filename, e.outputDir) {
		return filename
	}
	return path.Clean(e.outputDir + "/" + filename)
}

//__________________________________________________________________ set/get

// Sets the FD grid size
func (e *Engine) SetGridSize(size3D []int) {
	Debug("Engine.SetGridSize", size3D)
	Assert(len(size3D) == 3)
	if e.size3D == nil {
		e.size3D = e.size3D_[:]
		copy(e.size3D, size3D)
	} else {
		panic(InputErr("Grid size already set"))
	}

	Log("Grid size:", Size(size3D), "Cells:", e.NCell())

	if !IsGoodGridSize(size3D) {
		Warn("Grid size", Size(size3D), "hurts performance. Each size should be 2^n * {1, 2, 5 or 7}. X,Y sizes should be >= 16.")
	}

	e.logTotalSize()
}

// Gets the FD grid size
func (e *Engine) GridSize() []int {
	if e.size3D == nil {
		panic(InputErr("Grid size should be set first"))
	}
	return e.size3D
}

// Returns the grid size after zero padding
func (e *Engine) PaddedSize() []int {
	return PadSize(e.size3D, e.periodic)
}

// Sets the FD cell size
func (e *Engine) SetCellSize(size []float64) {
	Debug("Engine.SetCellSize", size)
	Assert(len(size) == 3)
	if e.cellSize == nil {
		e.cellSize = e.cellSize_[:]
		copy(e.cellSize, size)
	} else {
		panic(InputErr("Cell size already set"))
	}

	Log("Cell size:", size[Z], "x", size[Y], "x", size[X], "m³")
	e.logTotalSize()
}

// log total size if known
func (e *Engine) logTotalSize() {
	if e.size3D != nil && e.cellSize != nil {
		size := e.WorldSize()
		Log("World size:", size[Z], "x", size[Y], "x", size[X], "m³")
	}
}

// Size of total simulated world, in meters.
func (e *Engine) WorldSize() []float64 {
	cell := e.CellSize()
	grid := e.GridSize()
	return []float64{float64(grid[X]) * cell[X], float64(grid[Y]) * cell[Y], float64(grid[Z]) * cell[Z]}
}

// Gets the FD cell size
func (e *Engine) CellSize() []float64 {
	if e.cellSize == nil {
		panic(InputErr("Cell size should be set first"))
	}
	return e.cellSize
}

// FD cell volume in m³
func (e *Engine) CellVolume() float64 {
	return e.cellSize_[0] * e.cellSize_[1] * e.cellSize_[2]
}

// Sets the periodicity in each dimension
func (e *Engine) SetPeriodic(p []int) {
	Debug("Engine.SetPeriodic", p)
	if e.set_periodic_ {
		panic(InputErr("Periodicity already set"))
	}
	Assert(len(p) == 3)
	copy(e.periodic, p)
	e.set_periodic_ = true

	Log("PBC:", e.periodic[Z], "x, ", e.periodic[Y], "x, ", e.periodic[X], "x")
}

// Gets the FD grid size
func (e *Engine) Periodic() []int {
	// OK if not yet set
	e.set_periodic_ = true // but should not be changed once used.
	return e.periodic
}

// Gets the total number of FD cells
func (e *Engine) NCell() int {
	return e.size3D_[0] * e.size3D_[1] * e.size3D_[2]
}

// Retrieve a quantity by its name.
// Lookup is case-independent
func (e *Engine) Quant(name string) *Quant {
	lname := strings.ToLower(name)
	if q, ok := e.quantity[lname]; ok {
		return q
	} else {
		e.addDerivedQuant(name)
		if q, ok := e.quantity[lname]; ok {
			return q
		} else {
			panic(Bug("engine.Quant()"))
		}
	}
	return nil //silence gc
}

// Returns whether a quantity is already defined in the engine.
func (e *Engine) HasQuant(name string) bool {
	lname := strings.ToLower(name)
	_, ok := e.quantity[lname]
	return ok
}

func (e *Engine) AddDeltaQuant(in, ref string) {
	if !e.HasQuant(in) {
		panic(InputErrF(in, "does not exist."))
	}
	if !e.HasQuant(ref) {
		panic(InputErrF(ref, "does not exist."))
	}
	qin := e.Quant(in)
	qref := e.Quant(ref)

	out := "Δ" + in

	qout := e.AddNewQuant(out, qin.nComp, qin.kind, qin.unit)
	qout.updater = NewΔUpdater(qin, qref, qout)
}

// Derived quantities are averages, components, etc. of existing quantities.
// They are added to the engine on-demand.
// Syntax:
//	"<q>"  		: average of q
//	"q.x"  		: x-component of q, must be vector
//	"q.xx" 		: xx-component of q, must be tensor
//	"<q.x>"		: average of x-component of q.
//  "fft(q)" 	: fft of q
func (e *Engine) addDerivedQuant(name string) {
	// fft
	if strings.HasPrefix(name, "fft(") && strings.HasSuffix(name, ")") {
		in := name[len("fft(") : len(name)-1]
		Debug(in)
		qin := e.Quant(in)
		if qin.kind == VALUE {
			panic(InputErrF(qin.Name(), "is not space-dependent, fft is meaningless."))
		}
		e.AddQuant(NewQuant(name, qin.nComp, gpu.FFTOutputSize(e.GridSize()), qin.kind, qin.unit, false, "fft of "+qin.desc))
		Debug(name)
		qout := e.Quant(name)
		qout.updater = NewFFTUpdater(qin, qout)
		return
	}
	// average
	if strings.HasPrefix(name, "<") && strings.HasSuffix(name, ">") {
		origname := name[1 : len(name)-1]
		original := e.Quant(origname)
		if original.kind == VALUE {
			panic(InputErrF(original.Name(), "is not space-dependent, can not take its average."))
		}
		e.AddNewQuant(name, original.nComp, VALUE, original.unit)
		derived := e.Quant(name)
		e.Depends(name, origname)
		derived.updater = NewAverageUpdater(original, derived)
		return
	}
	// component
	if strings.Contains(name, ".") {
		split := strings.Split(name, ".")
		if len(split) != 2 {
			e.panicNoSuchQuant(name)
			//panic(InputErr("engine: undefined quantity: " + name))
		}
		origname, compname := split[0], strings.ToLower(split[1])
		orig := e.Quant(origname)

		// parse component string ("X" -> 0)
		comp := -1
		ok := false
		switch orig.nComp {
		default:
			panic(InputErr(orig.Name() + " has no component " + compname))
		case 3:
			comp, ok = VectorIndex[strings.ToUpper(compname)]
			comp = SwapIndex(comp, 3)
		case 6:
			comp, ok = TensorIndex[strings.ToUpper(compname)]
			comp = SwapIndex(comp, 6)
		case 9:
			comp, ok = TensorIndex[strings.ToUpper(compname)]
			comp = SwapIndex(comp, 9)
		}
		if !ok {
			panic(InputErr("invalid component:" + compname))
		}

		derived := orig.Component(comp)
		derived.name = orig.name + "." + strings.ToLower(compname) // hack, graphviz can't handle "."
		e.AddQuant(derived)
		e.Depends(derived.name, origname)
		return
	}
	e.panicNoSuchQuant(name)
	//panic(InputErr("engine: undefined quantity: " + name))
}

func (e *Engine) panicNoSuchQuant(name string) {
	msg := " Undefined quantity: " + name
	msg += "\n Options are: ["
	for k, _ := range e.quantity {
		msg += " " + k
	}
	msg += "]"
	msg += "\n Names are case-independent"
	msg += "\n Spatial averages can be taken as <name>"
	msg += "\n Components can be taken as name.x, name.y, etc"
	panic(InputErr(msg))
}

//__________________________________________________________________ add

// Returns true if the named module is already loaded.
func (e *Engine) HasModule(name string) bool {
	for _, m := range e.modules {
		if m.Name == name {
			return true
		}
	}
	return false
}

// Low-level module load, not aware of dependencies
func (e *Engine) LoadModule(name string) {
	if e.size3D == nil {
		panic(InputErr("Grid size should be set before loading modules"))
	}
	if e.cellSize == nil {
		panic(InputErr("Cell size should be set before loading modules"))
	}

	if e.HasModule(name) {
		return
	}
	module := GetModule(name)
	Log("Loaded module", module.Name, ":", module.Description)
	module.LoadFunc(e)
	e.modules = append(e.modules, module)
}

// Low-level module load, not aware of dependencies
func (e *Engine) LoadModuleArgs(name string, ins, deps, outs []string) {
	if e.size3D == nil {
		panic(InputErr("Grid size should be set before loading modules"))
	}
	if e.cellSize == nil {
		panic(InputErr("Cell size should be set before loading modules"))
	}
	module := GetModule(name)
	Log("Loaded module", module.Name, ":", module.Description)
	Log("In: ", module.Args.InsMap, " Deps: ", module.Args.DepsMap, " Out: ", module.Args.OutsMap)

	args := GetParsedArgumentsMap(module, ins, deps, outs)
	module.LoadFunc(e, args)
	e.modules = append(e.modules, module)
}

// Constructs and adds an arbitrary quantity.
// (Also returns it, but it's not necessarily used further)
// Name tag is case-independent.
// TODO: refactor AddQuant(q*Quant)
// TODO: NewQuant should take size from global engine.
func (e *Engine) AddNewQuant(name string, nComp int, kind QuantKind, unit Unit, desc ...string) *Quant {
	const CPUONLY = false
	e.AddQuant(NewQuant(name, nComp, e.size3D, kind, unit, CPUONLY, desc...))
	return e.Quant(name)
}

// Add a quantity.
func (e *Engine) AddQuant(q *Quant) {
	lname := strings.ToLower(q.name)

	// quantity should not yet be defined
	if _, ok := e.quantity[lname]; ok {
		panic(Bug("engine: Already defined: " + q.name))
	}

	e.quantity[lname] = q
}

// Mark childQuantity to depend on parentQuantity.
// Multiply adding the same dependency has no effect.
func (e *Engine) Depends(childQuantity string, parentQuantities ...string) {
	child := e.Quant(childQuantity)
	for _, parentQuantity := range parentQuantities {
		parent := e.Quant(parentQuantity)

		for _, p := range child.parents {
			if p.name == parentQuantity {
				return // Dependency is already defined, do not add it twice
				//panic(Bug("Engine.addDependency(" + childQuantity + ", " + parentQuantity + "): already present"))
			}
		}

		child.parents[parent.Name()] = parent
		parent.children[child.Name()] = child
	}
}

// Add a 1st order partial differential equation:
//	d y / d t = diff
// E.g.: ODE1("m", "torque")
// No direct dependency should be declared between the arguments.
func (e *Engine) AddPDE1(y, diff string) {
	yQ := e.Quant(y)
	dQ := e.Quant(diff)

	// check that two solvers are not trying to update the same output quantity
	for _, eqn := range e.equation {
		for _, out := range eqn.output {
			if out.Name() == y {
				panic(Bug("Already output of an equation: " + y))
			}
		}
	}
	e.equation = append(e.equation, PDE1(yQ, dQ))
}

//________________________________________________________________________________ step

func (e *Engine) SetSolver(s Solver) {
	if e.solver != nil {
		panic(InputErr("solver already set"))
	}
	e.solver = s
}

// Takes one time step.
// It is the solver's responsibility to Update/Invalidate its dependencies as needed.
func (e *Engine) Step() {
	if len(e.equation) == 0 || e.solver == nil {
		// if no solvers are defined, just advance time.
		// yes, this can be the desired behavior.
		e.time.SetScalar(e.time.Scalar() + e.dt.Scalar())
		// Always update step last, signals completion of step
		e.step.SetScalar(e.step.Scalar() + 1)
	} else {
		e.solver.Step()
	}
	// notify that a step has been taken
	// check if output needs to be saved
	e.notifyAll()
}

// Takes N time steps
func (e *Engine) Steps(N int) {
	Log("Running", N, "steps.")
	for i := 0; i < N; i++ {
		e.Step()
		e.updateDash()
	}
	DashExit()
}

// Runs for a certain duration specified in seconds
func (e *Engine) Run(duration float64) {
	Log("Running for", duration, "s.")
	time := e.time
	start := time.Scalar()
	for time.Scalar() < (start + duration) {
		e.Step()
		e.updateDash()
	}
	DashExit()
}

// time of last dashboard update
var lastdash int64

// refresh dashboard every x nanoseconds
const UPDATE_DASH = 150 * 1e6

// INTERNAL: show live progress: steps, t, dt, outputID
func (e *Engine) updateDash() {
	t := Nanoseconds()
	if t-lastdash > UPDATE_DASH {
		lastdash = t
		Dashboard(" step", e.step.multiplier[0],
			"t:", float64(e.time.multiplier[0]), "s",
			"dt:", float64(e.dt.multiplier[0]), "s",
			"out:", e._outputID)
	}
}

//__________________________________________________________________ output

// Notifies all crontabs that a step has been taken.
func (e *Engine) notifyAll() {
	for _, tab := range e.crontabs {
		tab.Notify(e)
	}
}

// Saves the quantity once in the specified format and file name
func (e *Engine) SaveAs(q *Quant, format string, options []string, filename string) {
	q.Update() //!!
	checkKinds(q, MASK, FIELD)
	out := OpenWRONLY(e.Relative(filename))
	defer out.Close()
	bufout := Buffer(out)
	defer bufout.Flush()
	GetOutputFormat(format).Write(bufout, q, options)
}

// Append the quantity once in the specified format and file name
func (e *Engine) SaveAsAppend(q *Quant, format string, options []string, filename string) {
	q.Update() //!!
	checkKinds(q, MASK, FIELD)
	out := OpenWRAPPENDONLY(e.Relative(filename))
	defer out.Close()
	bufout := Buffer(out)
	defer bufout.Flush()
	GetOutputFormat(format).Write(bufout, q, options)
}

// Saves the quantity periodically.
func (e *Engine) AutoSave(quant string, format string, options []string, period float64) (handle int) {
	checkKinds(e.Quant(quant), MASK, FIELD)
	handle = e.NewHandle()
	e.crontabs[handle] = &AutoSave{quant, format, options, period, e.time.Scalar(), 0}
	Debug("Auto-save", quant, "every", period, "s", "(handle ", handle, ")")
	return handle
}

// See api.go
func (e *Engine) Tabulate(quants []string, filename string) {
	if _, ok := e.outputTables[filename]; !ok { // table not yet open
		e.outputTables[filename] = NewTable(e.Relative(filename))
	}
	table := e.outputTables[filename]
	for _, q := range quants {
		e.Quant(q).Update() //!
	}
	table.Tabulate(quants)
}

// See api.go
func (e *Engine) AutoTabulate(quants []string, filename string, period float64) (handle int) {
	for _, q := range quants {
		checkKinds(e.Quant(q), MASK, VALUE)
	}
	handle = e.NewHandle()
	e.crontabs[handle] = &AutoTabulate{quants, filename, period, 0}
	Debug("Auto-tabulate", quants, "every", period, "s", "(handle ", handle, ")")
	return handle
}

// Generates an automatic file name for the quantity, given the output format.
// E.g., "dir.out/m000007.omf"
// see: outputDir, filenameFormat
func (e *Engine) AutoFilename(quant, format string) string {
	filenum := fmt.Sprintf(e.filenameFormat, e.OutputID())
	filename := quant + filenum + "." + GetOutputFormat(format).Name()
	dir := ""
	if e.outputDir != "" {
		dir = e.outputDir + "/"
	}
	return dir + filename
}

// Looks for an object with the handle number and removes it.
// Currently only looks in the crontabs.
func (e *Engine) RemoveHandle(handle int) {
	found := false
	if _, ok := e.crontabs[handle]; ok {
		delete(e.crontabs, handle)
		found = true
	}
	// TODO: if handles are used by other objects than crontabs, find them here
	if !found {
		Log(e.crontabs)
		panic(IOErr(fmt.Sprint("handle does not exist:", handle)))
	}
}

// INTERNAL: Used by frontend to set the output dir
func (e *Engine) SetOutputDirectory(dir string) {
	e.outputDir = dir
}

// String representation
func (e *Engine) String() string {
	str := "engine\n"
	quants := e.quantity
	for k, v := range quants {
		str += "\t" + k + "("
		for _, p := range v.parents {
			str += p.name + " "
		}
		str += ")\n"
	}
	//str += "ODEs:\n"
	//for _, ode := range e.ode {
	//	str += "d " + ode[0].Name() + " / d t = " + ode[1].Name() + "\n"
	//}
	return str
}

// DEBUG: statistics
func (e *Engine) Stats() string {
	str := fmt.Sprintln("engine running", e.timer.Seconds(), "s")
	for _, eqn := range e.equation {
		str += fmt.Sprintln(eqn.String())
	}
	quants := e.quantity
	for _, v := range quants {
		gpu := "     "
		if v.cpuOnly {
			gpu = "[CPU]"
		}
		str += fmt.Sprintln(fill(v.Name()), "\t", gpu,
			valid(v.upToDate), " upd:", fill(v.updates),
			" inv:", fill(v.invalidates),
			valid(v.bufUpToDate), " xfer:", fill(v.bufXfers),
			" ", fmt.Sprintf("%f", v.timer.Average()*1000), "ms/upd ",
			v.multiplier, v.unit)
	}
	return str
}

func (e *Engine) SaveState(out, in string) {
	if !e.HasQuant(in) {
		panic(InputErrF(in, "does not exist."))
	}

	qin := e.Quant(in)
	qin.Update() //!!!!!

	if !e.HasQuant(out) {
		e.AddNewQuant(out, qin.NComp(), qin.Kind(), qin.Unit(), qin.desc)
	}

	qout := e.Quant(out)
	qout.CopyFromQuant(qin)
}

func (e *Engine) RecoverState(out, in string) {
	if !e.HasQuant(out) {
		panic(InputErrF(out, "does not exist."))
	}
	qout := e.Quant(out)
	if !e.HasQuant(in) {
		panic(InputErrF(in, "does not exist."))
	}
	qin := e.Quant(in)
	qout.CopyFromQuant(qin)
}

func (e *Engine) UpdateEqRHS() {
	for _, eq := range e.equation {
		eq.UpdateRHS()
	}
}

func valid(b bool) string {
	if b {
		return "OK"
	}
	return "X"
}

func fill(s interface{}) string {
	str := fmt.Sprint(s)
	for len(str) < 8 {
		str += " "
	}
	return str
}
