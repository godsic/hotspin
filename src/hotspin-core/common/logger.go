//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// This file implements a global logger that prints to screen (stderr) and to a log file.
// The screen output can be filtered/disabled, while all output always goes to the log file.
// Author: Arne Vansteenkiste

import (
	"fmt"
	"log"
	"os"
	"sync"
	"time"
)

// Global debug flag.
// Typical use: if DEBUG {Debug(...)}
var DEBUG bool = true

// INTERNAL global logger
var logger Logger

func init() {
	InitLogger("") // make sure there is always *a* logger present. May be re-initiated later.
}

// INTERNAL
type Logger struct {
	ShowDebug   bool        // Include debug messages in stderr output?
	ShowWarn    bool        // Include warnings in stderr output?
	ShowPrint   bool        // Include normal output in stderr output?
	Screen      *log.Logger // Logs to the screen (stderr), usually prints only limited output
	File        *log.Logger // Logs to a log file, usually prints all output (including debug)
	Initialized bool        // If the logger is not initialized, dump output to stderr.
}

// Initiates the logger and sets the log file.
// logfile == "" disables logging to file.
func InitLogger(logfile string, options ...LogOption) {
	logger.Init(logfile, options...)
}

type LogOption int

const (
	LOG_NOSTDOUT LogOption = 1 << iota
	LOG_NODEBUG  LogOption = 1 << iota
	LOG_NOWARN   LogOption = 1 << iota
)

const (
	LOG_DEBUG_COL  = GREEN
	LOG_WARN_COL   = RED
	LOG_ERR_COL    = BOLD + RED
	LOG_NORMAL_COL = RESET
)

// INTERNAL Initiates the logger and sets a log file.
func (l *Logger) Init(logfile string, options ...LogOption) {
	opt := 0
	for i := range options {
		opt |= int(options[i])
	}
	l.Screen = log.New(os.Stderr, "", 0)

	l.ShowDebug = opt&int(LOG_NODEBUG) == 0
	l.ShowWarn = opt&int(LOG_NOWARN) == 0
	l.ShowPrint = opt&int(LOG_NOSTDOUT) == 0
	if logfile != "" {
		out, err := os.Create(logfile)
		CheckErr(err, ERR_IO)
		l.File = log.New(out, "", log.Ltime|log.Lmicroseconds)
		//Debug("Opened log file:", logfile)
	}
	l.Initialized = true
}

// Log a debug message.
func Debug(msg ...interface{}) {
	//if !DEBUG {panic(Bug("Need to put if DEBUG{...}")}
	// put [debug] in front of message
	msgcol := append([]interface{}{LOG_DEBUG_COL + "[debug]"}, msg...)
	msgcol = append(msgcol, RESET)
	msg = append([]interface{}{"[debug]"}, msg...)

	LogFile(msg...)

	if !logger.Initialized && logger.ShowDebug {
		fmt.Fprintln(os.Stderr, msgcol...)
	}
	if logger.ShowDebug {
		debugHook()
		logger.Screen.Println(msgcol...)
		debugHook()
	}
}

const MSG_WARNING = "Warning:"

// Log a warning.
func Warn(msg ...interface{}) {
	// put [warn ] in front of message
	msgcol := append([]interface{}{LOG_WARN_COL + "[warn ]"}, msg...)
	msgcol = append(msgcol, RESET)
	msg = append([]interface{}{"[warn ]"}, msg...)

	if !logger.Initialized && logger.ShowWarn {
		fmt.Fprintln(os.Stderr, msgcol...)
	}
	if logger.ShowWarn {
		logger.Screen.Println(msgcol...)
	}
	LogFile(msg...)
}

// Log an Error.
func Err(msg ...interface{}) {
	// put [err ] in front of message
	msgcol := append([]interface{}{LOG_ERR_COL + "[error]"}, msg...)
	msgcol = append(msgcol, RESET)
	msg = append([]interface{}{"[error]"}, msg...)

	if !logger.Initialized {
		fmt.Fprintln(os.Stderr, msgcol...)
	}
	logger.Screen.Println(msgcol...)
	LogFile(msg...)
}

// Log normal output.
func Log(msg ...interface{}) {
	// put [log  ] in front of message
	msgcol := append([]interface{}{LOG_NORMAL_COL + ""}, msg...) // no [log] on screen, only in file
	msg = append([]interface{}{"[log  ]"}, msg...)

	if !logger.Initialized {
		fmt.Fprintln(os.Stderr, msgcol...)
	}
	if logger.ShowPrint {
		logger.Screen.Println(msgcol...)
	}
	LogFile(msg...)
}

// Log to the log file only.
func LogFile(msg ...interface{}) {
	if logger.File != nil {
		logger.File.Println(msg...)
	}
}

// Used for showing "live" progress in the terminal.
// Prints the message but does not move the cursor down.
// The next call will replace the previous line.
// To resume normal printing, call DashExit() once.
func Dashboard(msg ...interface{}) {
	if logger.ShowPrint {
		fmt.Print(ERASE)
		fmt.Println(msg...)
		fmt.Print(LINEUP)
	}
}

// Resume normal printing after previous Dashboard() calls.
func DashExit() {
	fmt.Print(LINEDOWN)
}

func nop() {}

var debugHook func() = nop

// DEBUG: calls f before and after every Debug()
func SetDebugHook(f func()) {
	debugHook = f
}

// Hack to avoid cyclic dependency on engine.
var (
	progress_ func(int, int, string) = PrintProgress
	progLock  sync.Mutex
)

// Set progress bar to progress/total and display msg
// if GUI is up and running.
func Progress(progress, total int, msg string) {
	progLock.Lock()
	defer progLock.Unlock()
	if progress_ != nil {
		progress_(progress, total, msg)
	}
}

var (
	lastPct   = -1      // last progress percentage shown
	lastProgT time.Time // last time we showed progress percentage
)

func PrintProgress(prog, total int, msg string) {
	pct := (prog * 100) / total
	if pct != lastPct { // only print percentage if changed
		if (time.Since(lastProgT) > time.Second) || pct == 100 { // only print percentage once/second unless finished
			Debug(msg, pct, "%")
			lastPct = pct
			lastProgT = time.Now()
		}
	}
}

// Sets the function to be used internally by Progress.
// Avoids cyclic dependency on engine.
func SetProgress(f func(int, int, string)) {
	progLock.Lock()
	defer progLock.Unlock()
	progress_ = f
}
