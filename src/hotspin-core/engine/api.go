//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// The methods in this file will be automatically exposed in the mumax API
// through the apigen program.
//
//	NOTE: Here the user input (X,Y,Z) is changed to internal input (Z,Y,X)

import (
	"fmt"
	. "hotspin-core/common"
	"hotspin-core/host"
	"os"
	"path"
	"reflect"
	"runtime"
)

const (
	FINE_T     = 100  // subdivision of time interval in dispersion calculation
	B_EXT      = 1e-4 // the amplitude of excitation used in dispersion calculation
	OUT_FORMAT = "dump"
	N_TC       = 24
)

// The API methods are accessible to the end-user through scripting languages.
type API struct {
	Engine *Engine
}

//________________________________________________________________________________ init

// Set the grid size.
// WARNING: convert to ZYX
func (a API) SetGridSize(x, y, z int) {
	a.Engine.SetGridSize([]int{z, y, x}) // convert to internal axes
}

// Get the grid size.
// WARNING: convert to ZYX
func (a API) GetGridSize() (x, y, z int) {
	size := a.Engine.GridSize()
	return size[Z], size[Y], size[X] // convert to internal axes
}

// Set the cell size.
// WARNING: convert to ZYX
func (a API) SetCellSize(x, y, z float64) {
	a.Engine.SetCellSize([]float64{z, y, x}) // convert to internal axes and units
}

// Get the cell size.
// WARNING: convert to ZYX, internal units
func (a API) GetCellSize() (x, y, z float64) {
	size := a.Engine.CellSize()
	return size[Z], size[Y], size[X] // convert to internal axes
}

// Get the toal size, in meters, of the simulated world.
func (a API) GetWorldSize() (x, y, z float64) {
	size := a.Engine.WorldSize()
	return size[Z], size[Y], size[X] // convert to internal axes
}

// Set periodic boundary conditions in each direction.
// A value of 0 means no periodicity in that direction (the default).
// A nonzero value means the system is infinitely reproduced in that direction.
// The magnitude of the nonzero value is a hint of how accurately the
// infinite character should be approached, if applicable.
// E.g.: for the ferromagnetic exchange interaction,
// any nonzero value will give the same result: perfect infinite periodicity.
// But for the magnetostatic interaction, the magnitude of the nonzero value
// may be used as a hint where to cut off the magnetic field.
func (a API) SetPeriodic(x, y, z int) {
	a.Engine.SetPeriodic([]int{z, y, x})
}

// Get the periodicity
// WARNING: convert to ZYX, internal units
func (a API) GetPeriodic() (x, y, z int) {
	p := a.Engine.Periodic()
	return p[Z], p[Y], p[X] // convert to internal axes
}

// Load a physics module.
func (a API) Load(name string) {
	a.Engine.LoadModule(name)
}

// Load a physics module with user-defined variables.
func (a API) LoadArgs(name string, ins, deps, outs []string) {
	a.Engine.LoadModuleArgs(name, ins, deps, outs)
}

//________________________________________________________________________________ run

// Take one solver step
func (a API) Step() {
	a.Engine.Step()
}

// Takes N solver steps
func (a API) Steps(N int) {
	a.Engine.Steps(N)
}

// Runs for a duration given in seconds.
// TODO: precise stopping time
func (a API) Run(duration float64) {
	a.Engine.Run(duration)
}

// Runs the simulation until quantity a < value
func (a API) Run_Until_Smaller(quantity string, value float64) {
	e := a.Engine
	q := e.Quant(quantity)
	Log("Running until", q.Name(), "<", value, q.Unit())
	for q.Scalar() >= value {
		e.Step()
		e.updateDash()
	}
	DashExit()
}

// Runs the simulation until quantity a > quantity b
func (a API) Run_Until_Larger(quantity string, value float64) {
	e := a.Engine
	q := e.Quant(quantity)
	Log("Running until", q.Name(), ">", value, q.Unit())
	for q.Scalar() <= value {
		e.Step()
		e.updateDash()
	}
	DashExit()
}

//________________________________________________________________________________ set quantities

// Set value of a quantity. The quantity must be of type VALUE or MASK.
// If the quantity is a MASK, the value will be multiplied by a space-dependent mask
// which typically contains dimensionless numbers between 0 and 1.
func (a API) SetV(quantity string, value []float64) {
	q := a.Engine.Quant(quantity)
	SwapXYZ(value)
	q.SetValue(value)
}

func (a API) SetVMap(yName, xName string, yValue [][]float64, xValue []float64) {
	e := a.Engine

	x := e.Quant(xName)
	y := e.Quant(yName)

	if x.NComp() > 1 {
		panic(InputErrF("Can not map quantity", yName, " to non-scalar quantity", xName))
	}

	u := y.GetUpdater()
	if u == nil {
		u = newPointwiseOfUpdater(x, y)
		y.SetUpdater(u)
	}

	pointwise, ok := u.(*PointwiseOfUpdater)
	if !ok {
		panic(InputErrF("Can not set quantity", yName, " as a function of", xName, " since it is already determined in an other way:", reflect.TypeOf(u)))
	}

	pointwise.AppendMap(xValue, yValue)

	y.Invalidate()
}

// Set scalar. Convenience method for SetValue() with only one number.
// REDUNDANT?
func (a API) SetS(quantity string, value float64) {
	q := a.Engine.Quant(quantity)
	q.SetValue([]float64{value})
}

// Sets a space-dependent multiplier mask for the quantity.
// The value of the quantity (set by SetValue), will be multiplied
// by the mask value in each point of space. The mask is dimensionless
// and typically contains values between 0 and 1.
func (a API) SetMask(quantity string, mask *host.Array) {
	q := a.Engine.Quant(quantity)
	qArray := q.Array()
	if !EqualSize(mask.Size3D, qArray.Size3D()) {
		Log("Auto-resampling ", q.Name(), "from", Size(mask.Size3D), "to", Size(qArray.Size3D()))
		mask = Resample(mask, qArray.Size3D())
	}
	q.SetMask(mask)
}

// Like SetMask but reads the mask from a file.
func (a API) SetMask_File(quantity string, filename string) {
	a.SetMask(quantity, ReadFile(filename))
}

// Sets a space-dependent field quantity, like the magnetization.
func (a API) SetArray(quantity string, field *host.Array) {
	q := a.Engine.Quant(quantity)
	qArray := q.Array()
	if !EqualSize(field.Size3D, qArray.Size3D()) {
		Log("Auto-resampling ", quantity, "from", Size(field.Size3D), "to", Size(qArray.Size3D()))
		field = Resample(field, qArray.Size3D())
	}
	// setting a field when there is a non-1 multiplier is too confusing to allow
	for _, m := range q.multiplier {
		if m != 1 {
			panic(InputErr(fmt.Sprint(q.Name(), " is not an oridinary array, but has a mask + multiplier value. Did you mean to set the mask or the multiplier instead of the array?")))
		}
	}
	q.SetField(field)
}

// Like SetArray but reads the array from a file.
func (a API) SetArray_File(quantity string, filename string) {
	a.SetArray(quantity, ReadFile(filename))
}

//________________________________________________________________________________ get quantities

// Get the value of a space-independent or masked quantity.
// Returns an array with vector components or an
// array with just one element in case of a scalar quantity.
func (a API) GetV(quantity string) []float64 {
	q := a.Engine.Quant(quantity)
	q.Update() //!
	value := make([]float64, len(q.multiplier))
	copy(value, q.multiplier)
	SwapXYZ(value)
	return value
}

// DEBUG: Does not update.
func (a API) DebugV(quantity string) []float64 {
	q := a.Engine.Quant(quantity)
	//q.Update() //!
	value := make([]float64, len(q.multiplier))
	copy(value, q.multiplier)
	SwapXYZ(value)
	return value
}

// Gets the quantities unit.
func (a API) Unit(quantity string) string {
	return string(a.Engine.Quant(quantity).unit)
}

// Get the value of a scalar, space-independent quantity.
// Similar to GetValue, but returns a single number.
func (a API) GetS(quantity string) float64 {
	q := a.Engine.Quant(quantity)
	q.Update() //!
	return q.Scalar()
}

// Gets a space-dependent quantity. If the quantity uses a mask,
// the result is equal to GetMask() * GetValue()
func (a API) GetArray(quantity string) *host.Array {
	q := a.Engine.Quant(quantity)
	checkKinds(q, MASK, FIELD)
	q.Update() //!
	return q.Buffer(FIELD)
}

// Gets a space-dependent quantity. If the quantity uses a mask,
// the result is equal to GetMask() * GetValue()
func (a API) GetMask(quantity string) *host.Array {
	q := a.Engine.Quant(quantity)
	checkKinds(q, MASK, FIELD)
	q.Update() //!
	return q.Buffer(MASK)
}

// DEBUG: does not update
func (a API) DebugField(quantity string) *host.Array {
	q := a.Engine.Quant(quantity)
	checkKinds(q, MASK, FIELD)
	//q.Update() //!
	buffer := q.Buffer(FIELD)
	return buffer
}

// FOR DEBUG ONLY.
// Gets the quantity's array, raw.
func (a API) Debug_GetArray(quant string) *host.Array {
	q := a.Engine.Quant(quant)
	q.Update() //!
	array := q.Array()
	buffer := q.Buffer(FIELD)
	array.CopyToHost(buffer)
	return buffer
}

// Gets the value of the quantity at cell position x,y,z
func (a API) GetCell(quant string, x, y, z int) []float64 {
	q := a.Engine.Quant(quant)
	q.Update() //!
	value := make([]float64, q.NComp())
	if q.Array().IsNil() {
		for c := range value {
			value[c] = q.multiplier[c]
		}
	} else {
		for c := range value {
			value[c] = q.multiplier[c] * float64(q.Array().Get(c, z, y, x))
		}
	}
	SwapXYZ(value)
	return value
}

// Sets the value of the quantity at cell position x,y,z
func (a API) SetCell(quant string, x, y, z int, value []float64) {
	q := a.Engine.Quant(quant)
	SwapXYZ(value)
	pointer := q.Array().DevicePtr()
	if pointer == 0 {
		panic(InputErr(fmt.Sprint("You are attempting to call setcell() before array for qunatity is allocated. Please consider setmask() instead.")))
	}
	//q.assureAlloc()
	for c := range value {
		q.Array().Set(c, z, y, x, float64(value[c]))
	}
	q.Invalidate() //!
}

// ________________________________________________________________________________ save quantities

// Saves a space-dependent quantity, once. Uses the specified format and gives an automatic file name (like "m000001.png").
// See also: Save()
func (a API) Save(quantity string, format string, options []string) {
	quant := a.Engine.Quant(quantity)
	filename := a.Engine.AutoFilename(quantity, format)
	a.Engine.SaveAs(quant, format, options, filename)
}

// Saves a space-dependent quantity, once. Uses the specified format and file name.
func (a API) SaveAs(quantity string, format string, options []string, filename string) {
	a.Engine.SaveAs(a.Engine.Quant(quantity), format, options, filename)
}

// Saves a space-dependent quantity periodically, every period (expressed in seconds).
// Output appears in the output directory with automatically generated file names.
// E.g., for a quantity named "m", and format "txt" the generated files will be:
//	m00000.txt m00001.txt m00002.txt...
// See FilenameFormat() for setting the number of zeros.
// Returns an integer handle that can be used to manipulate the auto-save entry.
// E.g. remove(handle) stops auto-saving it.
// @see filenumberfomat
func (a API) AutoSave(quantity string, format string, options []string, period float64) (handle int) {
	return a.Engine.AutoSave(quantity, format, options, period)
}

// Saves these space-independent quantities, once.
// Their values are appended to the file, on one line.
func (a API) Tabulate(quantities []string, filename string) {
	a.Engine.Tabulate(quantities, filename)
}

// Saves any number of space-independent quantities periodically,
// every period (expressed in seconds).
// The values are appended to the file.
// Returns an integer handle that can be used to manipulate the auto-save entry.
// E.g. remove(handle) stops auto-saving it.
func (a API) AutoTabulate(quantities []string, filename string, period float64) (handle int) {
	return a.Engine.AutoTabulate(quantities, filename, period)
}

// Removes the object with given handle.
// E.g.:
//	handle = autosave(...)
//	remove(handle) # stops auto-saving
func (a API) Remove(handle int) {
	a.Engine.RemoveHandle(handle)
}

// Sets a global C-style printf format string used to generate file names for automatically saved files.
// The default "%06d" generates, e.g., "m000001.txt". "%d" would generate, e.g., "m1.txt".
func (a API) FileNumberFormat(format string) {
	a.Engine.filenameFormat = format
	Log("Using", format, "to number automatically saved files.")
}

// Returns the output directory for the running simulation.
func (a API) OutputDirectory() string {
	return a.Engine.outputDir
}

//________________________________________________________________________________ add quantities

// Add a new quantity to the multi-physics engine, its
// value is added to the (existing) sumQuantity.
// E.g.: Add_To("H", "H_1") adds a new external field
// H_1 that will be added to H.
func (a API) Add_To(sumQuantity, newQuantity string) {
	a.Add_To_Weighted(sumQuantity, newQuantity, 1.0)
}

// Add a new quantity to the multi-physics engine, its
// value is multiplied by the weight value and add to the (existing) sumQuantity.
// E.g.: Add_To_Weight("H", "H_1") adds a new external field
// H_1 that will be added to H.

func (a API) Add_To_Weighted(sumQuantity, newQuantity string, weight float64) {

	e := a.Engine
	sumQuant := e.Quant(sumQuantity)
	sumUpd, ok := sumQuant.GetUpdater().(*SumUpdater)
	if !ok {
		panic(InputErrF("Add_To: quantity ", sumQuant.Name(), " is not of type 'sum', nothing can be added to it."))
	}

	var term *Quant

	if e.HasQuant(newQuantity) {
		term = e.Quant(newQuantity)
	} else {
		term = e.AddNewQuant(newQuantity, sumQuant.NComp(), MASK, sumQuant.Unit())
	}

	sumUpd.MAddParent(term.Name(), weight)
	Log("Added new quantity", term.FullName(), "to", sumQuant.Name())

	//e := a.Engine
	//sumQuant := e.Quant(sumQuantity)
	//term := e.AddNewQuant(newQuantity, sumQuant.NComp(), MASK, sumQuant.Unit())
	//AddTermToQuant(sumQuant, term)
}

// untested
//func (a API) Add_Dot(dotQuantity, inputQuant1, inputQuant2 string) {
//	e := a.Engine
//	inQuant1 := e.Quant(inputQuant1)
//	inQuant2 := e.Quant(inputQuant2)
//	unit := "(" + inQuant1.Unit() + ")*(" + inQuant2.Unit() + ")"
//	dotQuant := e.AddNewQuant(dotQuantity, SCALAR, VALUE, unit)
//	dotQuant.SetUpdater(NewSDotUpdater(dotQuant, inQuant1, inQuant2, 1))
//	Log("Added new dot product:", dotQuantity, "=", inputQuant1, "·", inputQuant2)
//}

// Add a new quantity to the multi-physics engine, its
// value is the maximum of the absolute value of inputQuantity.
// E.g.: New_MaxAbs("max_torque", "torque") adds a new quantity
// "max_torque" whose value is max(abs(torque)). For vector
// quantities, the maximum is taken over all components.
func (a API) New_MaxAbs(newQuantity, inputQuantity string) {
	e := a.Engine
	In := e.Quant(inputQuantity)
	checkKind(In, FIELD)
	New := e.AddNewQuant(newQuantity, SCALAR, VALUE, In.Unit())
	New.SetUpdater(NewMaxAbsUpdater(In, New)) // also sets dependency
}

// Add a new quantity to the multi-physics engine, its
// value is the maximum norm of inputQuantity (a 3-component vector).
// E.g.: New_MaxNorm("maxnorm_torque", "torque") adds a new quantity
// "maxnorm_torque" whose value is max(norm(torque)).
func (a API) New_MaxNorm(newQuantity, inputQuantity string) {
	e := a.Engine
	In := e.Quant(inputQuantity)
	checkKind(In, FIELD)
	checkComp(In, 3)
	New := e.AddNewQuant(newQuantity, SCALAR, VALUE, In.Unit())
	New.SetUpdater(NewMaxNormUpdater(In, New)) // also sets dependency
}

func (a API) New_Peak(newQuantity, inputQuantity string) {
	e := a.Engine
	In := e.Quant(inputQuantity)
	checkKind(In, VALUE)
	checkComp(In, 1)
	New := e.AddNewQuant(newQuantity, SCALAR, VALUE, In.Unit())
	New.SetUpdater(NewPeakUpdater(In, New))
}

//________________________________________________________________________________ misc

// Saves an image file of the physics graph using the given file name.
// The extension determines the output format. E.g.: .png, .svg, ...
// A file with a .dot extension will be written as well.
// Rendering requires the package "graphviz" to be installed.
func (a API) SaveGraph(file string) {

	file = a.Engine.Relative(file)
	dotfile := ReplaceExt(file, ".dot")

	f, err := os.Create(dotfile)
	defer f.Close()
	CheckIO(err)
	a.Engine.WriteDot(f)
	Log("Wrote", dotfile)
	RunDot(dotfile, path.Ext(file)[1:]) // rm .
}

// DEBUG
func (a API) PrintStats() {
	Log(a.Engine.Stats())
}

// DEBUG: manually update the quantity state
func (a API) Debug_Update(quantity string) {
	a.Engine.Quant(quantity).Update()
}

// DEBUG: manually update the quantity state
func (a API) Debug_Invalidate(quantity string) {
	a.Engine.Quant(quantity).Invalidate()
}

// DEBUG: removes the updater of this quantity
func (a API) Debug_DisableUpdate(quantity string) {
	a.Engine.Quant(quantity).updater = nil
}

// DEBUG: verify all quanties' values
func (a API) Debug_VerifyAll() {
	e := a.Engine
	for _, q := range e.quantity {
		q.Verify()
	}
}

func (a API) Debug_GC() {
	Log("GC")
	runtime.GC()
}

// DEBUG: start a timer with a given identifier tag
func (a API) StartTimer(tag string) {
	EnableTimers(true)
	Start(tag)
}

// DEBUG: stop a timer with a given identifier tag.
// It must be started first.
func (a API) StopTimer(tag string) {
	Stop(tag)
}

// DEBUG: Gets the time, in seconds, recorded by the timer with this tag.
func (a API) GetTime(tag string) float64 {
	return GetTime(tag)
}

// DEBUG: Resets the timer with this tag.
func (a API) ResetTimer(tag string) {
	ResetTimer(tag)
}

// DEBUG: echos a string, can be used for synchronous output
func (a API) Echo(str string) {
	Log(str)
}

// Should be called at the end of each input file to make
// sure the input script does not terminate too early.
func (a API) Sync() {
	// The simple act of calling a function (any function)
	// causes the front-end RPC to synchronize the scripting
	// language and mumax.
	Log("Syncing")
}

// DEBUG: reads an array from a file.
func (a API) ReadFile(filename string) *host.Array {
	return ReadFile(filename)
}

// Returns the output ID corresponding to the current simulation time.
// All automatic output uses this number to identify the time corresponding
// to the saved quantity.
func (a API) OutputID() int {
	return a.Engine.OutputID()
}

func (a API) SaveState(dst, src string) {
	a.Engine.SaveState(dst, src)
}

func (a API) RecoverState(dst, src string) {
	a.Engine.RecoverState(dst, src)
}

func (a API) AddDeltaQuant(src, ref string) {
	a.Engine.AddDeltaQuant(src, ref)
}
