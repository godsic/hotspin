//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

import (
	"fmt"
	"time"
)

var (
	time0      time.Time
	time0_init bool
)

// Substitute for the old time.Nanoseconds().
// Does not start from Epoch so only suited to measure durations.
func Nanoseconds() int64 {
	if !time0_init {
		time0 = time.Now()
		time0_init = true
	}
	return int64(time.Now().Sub(time0))
}

// Non-thread safe timer for debugging.
// The zero value is usable without initialization.
type Timer struct {
	StartNanos, TotalNanos int64 // StartTime == 0: not running
	Count                  int
}

// Start the timer
func (t *Timer) Start() {
	if t.StartNanos != 0 {
		panic(Bug("Timer.Start: already running"))
	}
	t.StartNanos = Nanoseconds()
}

// Stop the timer
func (t *Timer) Stop() {
	if t.StartNanos == 0 {
		panic(Bug("Timer.Stop: not running"))
	}
	t.TotalNanos += (Nanoseconds() - t.StartNanos)
	t.Count++
	t.StartNanos = 0
}

// Returns the total number of seconds this timer has been running.
// Correct even if the timer is running when this function is called.
func (t *Timer) Seconds() float64 {
	if t.StartNanos == 0 { //not running for the moment
		return float64(t.TotalNanos) / 1e9
	} // running for the moment
	return float64(t.TotalNanos+Nanoseconds()-t.StartNanos) / 1e9
}

// Average number of seconds per call.
func (t *Timer) Average() float64 {
	return t.Seconds() / (float64(t.Count))
}

func (t *Timer) String() string {
	return fmt.Sprint(t.Count, "\t", float32(1000*t.Average()), " ms/call")
}

// Global timers indexed by a tag string
var timers map[string]Timer

// should we enable global timers?
var enableTimers bool

func init() {
	timers = make(map[string]Timer)
}

// enable/disable global timers
func EnableTimers(enable bool) {
	enableTimers = enable
}

// Start a global timer with tag name
func Start(tag string) {
	if !enableTimers {
		return
	}
	timer := timers[tag]
	timer.Start()       // usable zero value if timer was not yet defined
	timers[tag] = timer // have to write back modified value to map
}

// Stop a global timer with tag name
func Stop(tag string) {
	if !enableTimers {
		return
	}
	timer, ok := timers[tag]
	if !ok {
		panic(InputErr("Undefined timer: " + tag))
	}
	timer.Stop()
	timers[tag] = timer
}

// Re-set the timer to its initial state
func ResetTimer(tag string) {
	timer, ok := timers[tag]
	if !ok {
		panic(InputErr("Undefined timer: " + tag))
	}
	timer.StartNanos = 0
	timer.TotalNanos = 0
	timer.Count = 0
	timers[tag] = timer // have to write back modified value to map
}

// Get the total run time (in seconds) of a global timer
func GetTime(tag string) float64 {
	timer := timers[tag]
	return timer.Seconds()
}

// Print names and runtime of all global timers
func PrintTimers() {
	if !enableTimers || len(timers) == 0 {
		return
	}
	Debug(" ---- timers ----")
	for tag, timer := range timers {
		Debug(tag, ":\t", timer.String())
	}
	Debug(" ----------------")
}
