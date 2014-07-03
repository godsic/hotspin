//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package frontend

import (
	cu "cuda/driver"
	cuda "cuda/runtime"
	"flag"
	"fmt"
	. "hotspin-core/common"
	"hotspin-core/gpu"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
)

// command-line flags (more in engine/main.go)
var (
	flag_outputdir *string = flag.String("o", "", "Specify output directory")
	flag_force     *bool   = flag.Bool("f", false, "Force start, remove existing output directory")
	flag_logfile   *string = flag.String("log", "", "Specify log file")
	flag_command   *string = flag.String("command", "", "Override interpreter command")
	flag_debug     *bool   = flag.Bool("g", false, "Show debug output")
	flag_timing    *bool   = flag.Bool("t", false, "Enable timers for benchmarking")
	flag_cpuprof   *string = flag.String("cpuprof", "", "Write gopprof CPU profile to file")
	flag_memprof   *string = flag.String("memprof", "", "Write gopprof memory profile to file")
	flag_silent    *bool   = flag.Bool("s", false, "Be silent")
	flag_warn      *bool   = flag.Bool("w", true, "Show warnings")
	flag_help      *bool   = flag.Bool("h", false, "Print help and exit")
	flag_version   *bool   = flag.Bool("v", false, "Print version info and exit")
	flag_test      *bool   = flag.Bool("test", false, "Test CUDA and exit")
	flag_dontrun   *bool   = flag.Bool("dr", false, "Don't run mumax if the output directory exists")
	//flag_timeout   *string = flag.String("timeout", "", "Set a maximum run time. Units s,h,d are recognized.")
	flag_gpus  *string = flag.String("gpu", "0", "Which GPUs to use. gpu=0, gpu=0:3, gpu=1,2,3, gpu=all")
	flag_sched *string = flag.String("sched", "auto", "CUDA scheduling: auto|spin|yield|sync")
	flag_fft   *string = flag.String("fft", "X", "Override the FFT implementation (advanced)")
)

// Mumax2 main function
func Main() {
	// first test for flags that do not actually run a simulation
	flag.Parse()

	defer func() {
		cleanup()        // make sure we always clean up, no matter what
		err := recover() // if anything goes wrong, produce a nice crash report
		if err != nil {
			crashreport(err)
		}
	}()

	DEBUG = *flag_debug

	runtime.GOMAXPROCS(runtime.NumCPU())

	//initTimeout()

	if *flag_cpuprof != "" {
		f, err := os.Create(*flag_cpuprof)
		if err != nil {
			Log(err)
		}
		Log("Writing CPU profile to", *flag_cpuprof)
		pprof.StartCPUProfile(f)
		// will be flushed on cleanup
	}

	EnableTimers(*flag_timing)

	if *flag_help {
		fmt.Fprintln(os.Stderr, "Usage:")
		flag.PrintDefaults()
		return
	}

	if *flag_version {
		fmt.Println(WELCOME)
		fmt.Println("Go", runtime.Version())
		return
	}

	/*if flag.NArg() == 0 {
		Log("No input files, starting interactive mode")
		engineMain()
		return
	}*/

	if *flag_fft != "" {
		gpu.SetDefaultFFT(*flag_fft)
	}

	// else...
	clientMain()
}

// based on the -gpu flag, activate set of GPUs to use
func initMultiGPU() {
	flag := *flag_gpus
	cuFlags := parseCuFlags()

	gpus := Atoi(flag)
	gpu.InitGPU(gpus, cuFlags)
}

func parseCuFlags() uint {
	cudaflag := *flag_sched
	var flag uint

	switch cudaflag {
	default:
		panic(InputErr("Expecting auto,spin,yield or sync: " + cudaflag))
	case "auto":
		flag |= cu.CTX_SCHED_AUTO
	case "spin":
		flag |= cu.CTX_SCHED_SPIN
	case "yield":
		flag |= cu.CTX_SCHED_YIELD
	case "sync":
		flag |= cu.CTX_BLOCKING_SYNC
	}
	return flag
}

// return the input file. "" means none
func inputFile() string {
	// check if there is just one input file given on the command line
	if flag.NArg() == 0 {
		panic(InputErr("no input files"))
		return ""
	}
	if flag.NArg() > 1 {
		panic(InputErr(fmt.Sprint("need exactly 1 input file, but", flag.NArg(), "given:", flag.Args())))
	}
	return flag.Arg(0)
}

func cleanup() {
	Debug("cleanup")

	// write memory profile
	if *flag_memprof != "" {
		f, err := os.Create(*flag_memprof)
		if err != nil {
			Log(err)
		}
		Log("Writing memory profile to", *flag_memprof)
		pprof.WriteHeapProfile(f)
		f.Close()
	}

	// write cpu profile
	if *flag_cpuprof != "" {
		Log("Flushing CPU profile", *flag_cpuprof)
		pprof.StopCPUProfile()
	}

	// print timers
	if *flag_debug {
		PrintTimers()
	}

	cuda.DeviceReset()
	// kill subprocess?
}

func crashreport(err interface{}) {
	status := 0
	switch err.(type) {
	default:
		Err("panic:", err, "\n", getCrashStack())
		Log(SENDMAIL)
		status = ERR_PANIC
	case Bug:
		Err("bug:", err, "\n", getCrashStack())
		Log(SENDMAIL)
		status = ERR_BUG
	case InputErr:
		Err("illegal input:", err, "\n")
		if *flag_debug {
			Debug(getCrashStack())
		}
		status = ERR_INPUT
	case IOErr:
		Err("IO error:", err, "\n")
		if *flag_debug {
			Debug(getCrashStack())
		}
		status = ERR_IO
	case cu.Result:
		Err("cuda error:", err, "\n", getCrashStack())
		if *flag_debug {
			Debug(getCrashStack())
		}
		status = ERR_CUDA
	}
	Debug("Exiting with status", status, ErrString[status])
	os.Exit(status)
}

// Returns a stack trace for debugging a crash.
// The first irrelevant lines are discarded
// (they trace to this function), so the trace
// starts with the relevant panic() call.
func getCrashStack() string {
	stack := debug.Stack()
	return "stack trace:\n" + string(stack)
}

// sets up a timeout that will kill mumax when it runs too long
// TODO: does not work because Go runtime does not schedule round-robin.
//func initTimeout() {
//	timeout := *flag_timeout
//	t := 0.
//	if timeout != "" {
//		switch timeout[len(timeout)-1] {
//		default:
//			t = Atof64(timeout)
//		case 's':
//			t = Atof64(timeout[:len(timeout)-1])
//		case 'h':
//			t = 3600 * Atof64(timeout[:len(timeout)-1])
//		case 'd':
//			t = 24 * 3600 * Atof64(timeout[:len(timeout)-1])
//		}
//	}
//	if t != 0 {
//		Log("Timeout: ", t, "s")
//		go func() {
//			time.Sleep(int64(1e9 * t))
//			Log("Timeout reached:", timeout)
//			cleanup()
//			os.Exit(ERR_IO)
//		}()
//	}
//}

const (
	WELCOME = `
 hotspin v0.1.0114α by Mykola Dvornik (DyNaMat, Ghent University, Belgium)
`
	LOGFILE  = "hotspin.log"
	SENDMAIL = "\n------\nIf you believe this is a bug, please file an issue on https://github.com/godsic/hotspin/issues. Be sure to include the log file \"" + LOGFILE + "\n------\n"
)
