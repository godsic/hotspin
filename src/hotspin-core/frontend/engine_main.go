//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package frontend

// This file implements the mumax2's main routines for engine mode.

import (
	"bufio"
	cu "cuda/driver"
	"fmt"
	. "hotspin-core/common"
	"os"
	"runtime"
)

// run the input files given on the command line
func engineMain() {
	if !*flag_silent {
		fmt.Println(WELCOME)
	}

	outdir := "."
	if *flag_outputdir != "" {
		outdir = *flag_outputdir
	}
	initOutputDir(outdir)
	initLogger(outdir)
	LogFile(WELCOME)
	Debug("Go", runtime.Version())

	initCUDA()

	initMultiGPU()

	if *flag_test {
		testMain()
		return
	}

	var client Client
	client.Init("-", outdir)
	client.RunSlave()
}

func initCUDA() {
	Debug("Initializing CUDA")
	runtime.LockOSThread()
	Debug("Locked OS Thread")
	cu.Init(0)
}

// Do not start interpreter subprocess but wait for commands on Stdin.
// Used when mumax is the subprocess.
func (c *Client) RunSlave() {
	wflush := bufio.NewWriter(os.Stdout)
	c.ipc.Init(os.Stdin, os.Stdout, *wflush, c.api)
	c.ipc.Run()
}
