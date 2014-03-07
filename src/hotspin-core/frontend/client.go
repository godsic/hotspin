//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package frontend

// The client implements Inter-Process-Communication
// between mumax and a scripting language.
// Author: Arne Vansteenkiste

import (
	"bufio"
	"fmt"
	. "hotspin-core/common"
	"hotspin-core/engine"
	"net"
	"os"
	"os/exec"
	"path"
	"time"
)

type Client struct {
	inputFile     string
	outputDir     string
	commAddr      string
	ipc           jsonRPC
	api           engine.API
	wire          net.Conn
	isWireAlive   int
	server        net.Listener
	isServerAlive int
	logWait       chan int // channel to wait for completion of go logStream()
}

// Initializes the mumax client to parse infile, write output
// to outdir and connect to a server over conn.

func (c *Client) Init(inputfile string, outputDir string) {
	c.isServerAlive = 0
	c.isWireAlive = 0
	m_s, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		Debug("[WARNING] hotspin has failed to start slave server")
		return
	}
	c.server = m_s
	c.commAddr = c.server.Addr().String()
	Debug("The TCP connection is grangted to:", c.commAddr)
	c.isServerAlive = 1

	c.inputFile = inputfile
	c.outputDir = outputDir

	CheckErr(os.Setenv("MUMAX2_ADDR", c.commAddr), ERR_IO)

	engine.Init()
	engine.GetEngine().SetOutputDirectory(outputDir)
	c.api = engine.API{engine.GetEngine()}
}

// Start interpreter sub-command and communicate over fifos in the output dir.
func (c *Client) Run() {

	defer func() {
		Debug("Shutdown server...")
		if c.isWireAlive == 1 {
			err1 := c.wire.Close()
			CheckErr(err1, ERR_IO)
		}

		if c.isServerAlive == 1 {
			err2 := c.server.Close()
			CheckErr(err2, ERR_IO)
		}
	}()

	c.logWait = make(chan int)

	command, waiter := c.startSubcommand()

	swait := make(chan int)

	go func() {
		Debug("Waiting for client...")
		var err error
		c.wire, err = c.server.Accept()
		if err != nil {
			Debug("Client has failed to connect!")
			swait <- 1
			return
		}
		swait <- 0
	}()

	go func() {
		var status int
		status = <-waiter
		if status != 0 {
			swait <- status
			return
		}
		swait <- 0
	}()

	status := <-swait
	if status != 0 {
		Debug("Connection failed")
		panic(InputErr(fmt.Sprint(command, " exited with status ", status)))
		return
	}
	//c.wire = s_wire
	c.isWireAlive = 1

	s_infifo := bufio.NewReader(c.wire)
	s_outflush := bufio.NewWriter(c.wire)
	s_outfifo := s_outflush

	c.ipc.Init(s_infifo, s_outfifo, *s_outflush, c.api)
	c.ipc.Run()

	// wait for the sub-command to exit
	Debug("Waiting for subcommand ", command, "to exit")

	exitstat := <-swait

	if exitstat != 0 {
		panic(InputErr(fmt.Sprint(command, " exited with status ", exitstat)))
	}

	//Housekeeping

	c.wire.Close()
	c.isWireAlive = 0
	c.server.Close()
	c.isServerAlive = 0

	Debug("Client is now disconnected")

	// wait for full pipe of sub-command output to the logger
	// not sure if this has much effect.

	<-c.logWait // stderr
	<-c.logWait // stdout (or the other way around ;-)

}

// run the sub-command (e.g. python) to interpret the script file
// it will first hang while trying to open the FIFOs
func (c *Client) startSubcommand() (command string, waiter chan (int)) {

	CheckErr(os.Setenv("MUMAX2_OUTPUTDIR", c.outputDir), ERR_IO)

	var args []string
	command, args = commandForFile(c.inputFile) // e.g.: "python"
	Debug("Starting", command, "with following flags", args)
	proc := exec.Command(command, args...) //:= subprocess(command, args)
	stderr, err4 := proc.StderrPipe()
	CheckErr(err4, ERR_IO)
	stdout, err5 := proc.StdoutPipe()
	CheckErr(err5, ERR_IO)
	CheckErr(proc.Start(), ERR_IO)

	go logStream("["+command+"]", stderr, true, c.logWait)
	go logStream("["+command+"]", stdout, false, c.logWait)

	Debug(command, "PID:", proc.Process.Pid)

	// start waiting for sub-command asynchronously and
	// use a channel to signal sub-command completion
	waiter = make(chan (int))
	go func() {
		exitstat := 666 // dummy value
		err := proc.Wait()
		if err != nil {
			if msg, ok := err.(*exec.ExitError); ok {
				if msg.ProcessState.Success() {
					exitstat = 0
				} else {
					exitstat = 1
				}
				// TODO: extract unix exit status
				//exitstat = msg.ExitStatus()
			} else {
				panic(InputErr(err.Error()))
			}
		} else {
			exitstat = 0
		}
		waiter <- exitstat // send exit status to signal completion
	}()

	return
}

// given a file name (e.g. file.py)
// this returns a command to run the file (e.g. python file.py, java File)
func commandForFile(file string) (command string, args []string) {
	if *flag_command != "" {
		return *flag_command, []string{file}
	}
	if file == "" {
		panic(IOErr("no input file"))
	}
	switch path.Ext(file) {
	default:
		panic(InputErr("Cannot handle files with extension " + path.Ext(file)))
	case ".py":
		return "python", []string{file}
		//case ".java":
		//	return GetExecDir() + "javaint", []string{file}
		//case ".class":
		//	return "java", []string{ReplaceExt(file, "")}
		//case ".lua":
		//	return "lua", []string{file}
	}
	panic(Bug("unreachable"))
	return "", nil
}

// returns a channel that will signal when the file has appeared
func pollFile(fname string) (waiter chan (int)) {
	waiter = make(chan (int))
	go func() {
		for !FileExists(fname) {
			time.Sleep(10)
		}
		waiter <- 1
	}()
	return
}
