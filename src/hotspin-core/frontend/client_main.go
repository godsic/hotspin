//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package frontend

// This file implements the mumax2's main function.
// Arne Vansteenkiste

import (
	cu "cuda/driver"
	"fmt"
	. "hotspin-core/common"
	"io"
	"os"
	"path/filepath"
	"runtime"
)

// run the input files given on the command line
func clientMain() {
	//defer fmt.Println(RESET)

	if !*flag_silent {
		fmt.Println(WELCOME)
	}

	infile := inputFile()
	outdir := GetOutputDir(infile)

	initOutputDir(outdir)
	initLogger(outdir)

	LogFile(WELCOME)
	hostname, _ := os.Hostname()
	Debug("Hostname:", hostname)
	Debug("Go", runtime.Version())
	//command := *flag_command

	// initialize CUDA first
	Debug("Initializing CUDA")
	runtime.LockOSThread()
	Debug("Locked OS Thread")

	cu.Init(0)

	initMultiGPU()

	if *flag_test {
		testMain()
		return
	}

	ServeClient(infile, outdir)

}

// return the output directory
func GetOutputDir(inputFile string) string {
	if *flag_outputdir != "" {
		return *flag_outputdir
	}
	if inputFile == "" {
		return filepath.Dir(filepath.Clean(os.Args[0]))
	}
	return inputFile + ".out"
}

// make the output dir
func initOutputDir(outputDir string) {

	_, rooterr := os.Stat(outputDir)

	if *flag_force {
		if !os.IsNotExist(rooterr) {
			if outputDir != "." {
				err := filepath.Walk(outputDir, RemoveDirContent(outputDir))
				if err != nil {
					Log("filepath.Walk", outputDir, ":", err)

				}
			}
		}
	}

	if *flag_dontrun {
		errOut := os.Mkdir(outputDir, 0777)
		if outputDir != "." {
			CheckIO(errOut)
		} else {
			Log(errOut)
		}
	}

	_, rooterr1 := os.Stat(outputDir)
	if os.IsNotExist(rooterr1) {
		errOut := os.Mkdir(outputDir, 0777)
		if outputDir != "." {
			CheckIO(errOut)
		} else {
			Log(errOut)
		}
	}
}

func RemoveDirContent(outputDir string) filepath.WalkFunc {

	return func(path string, info os.FileInfo, err error) error {
		if path != outputDir {
			if info.IsDir() {
				err := os.RemoveAll(path)
				if err != nil {
					Log("os.RemoveAll", outputDir, ":", err)
				}
			} else {
				err := os.Remove(path)
				if err != nil {
					Log("os.Remove", outputDir, ":", err)
				}
			}
		}
		return nil
	}
}

// Gets the response from the client and starts the slave server
func ServeClient(inputfile string, ClientPath string) {

	var client Client
	client.Init(inputfile, ClientPath)
	client.Run()
}

// Gets the response from the client and starts the slave server
/*func ServeClient(wire net.Conn, masterctl chan int) {

	m_in := bufio.NewReader(wire)
	m_out := bufio.NewWriter(wire)

	// Reads Client's name and path to the script file if any

	cmsg, err := m_in.ReadString('\n')
	if err != nil {
		Debug("[WARNING] Client has failed to send its name and path to the script")
		return
	}

	cmsg_slice := strings.Split(cmsg,":")

	ClientName := cmsg_slice[0]
	// This approach is not universal at all
	ClientPath := strings.TrimSpace(cmsg_slice[1]) + ".out"

	if ClientName == "exit" {
		Debug("Exit request...")
		masterctl <- EXIT
		return

	}

	if ClientName == "terminate" {
		Debug("Termination request...")
		masterctl <- TERMINATE
		return
	}
	Debug(ClientName + " is connected")
	Debug("Client asks to write into: " + ClientPath)
	initOutputDir(ClientPath)

	s_ln, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		Debug("[WARNING] MuMax has failed to start slave server")
		return
	}

	s_addr	:= s_ln.Addr().String()
	Debug("The slave server is started on: " + s_addr)
	addr_msg := s_ln.Addr().String() + EOM
	m_out.WriteString(addr_msg)
	m_out.Flush()

	Debug("Waiting " + ClientName + " to respond to slave server...")
	s_wire, err := s_ln.Accept()
	if err != nil {
		Debug("[WARNING]" + ClientName + " has failed to connect")
		return
	}
	Debug("Done.")

	var client Client
	client.wire = s_wire
	client.Init(ClientPath)
	client.Run()
}*/

// return the output directory
func outputDir(inputFile string) string {
	if *flag_outputdir != "" {
		return *flag_outputdir
	}
	return inputFile + ".out"
}

// initialize the logger
func initLogger(outputDir string) {
	var opts LogOption
	if !*flag_debug {
		opts |= LOG_NODEBUG
	}
	if *flag_silent {
		opts |= LOG_NOSTDOUT | LOG_NODEBUG | LOG_NOWARN
	}
	if !*flag_warn {
		opts |= LOG_NOWARN
	}

	logFile := *flag_logfile
	if logFile == "" {
		logFile = outputDir + "/" + LOGFILENAME
	}
	InitLogger(logFile, opts)
	Debug("Logging to", logFile)
}

// IO buffer size
const BUFSIZE = 4096

// pipes standard output/err of the command to the logger
// typically called in a separate goroutine
func logStream(prefix string, in io.Reader, stderr bool, waiter chan int) {
	defer func() { waiter <- 1 }() // signal completion
	var bytes [BUFSIZE]byte
	buf := bytes[:]
	var err error = nil
	n := 0
	for err == nil {
		n, err = in.Read(buf)
		if n != 0 {
			if stderr {
				Err(prefix, string(buf[:n]))
			} else {
				Log(prefix, string(buf[:n]))
			}
		} // TODO: no printLN
	}
	Debug("logStream done: ", err)
}

const (
	INFIFO        = "in.fifo"   // FIFO filename for mumax->subprocess text-based function calls.
	OUTFIFO       = "out.fifo"  // FIFO filename for mumax<-subprocess text-based function calls.
	HANDSHAKEFILE = "handshake" // Presence of this file indicates subprocess initialization OK.
	PORT          = "0"
	SERVERADDR    = "localhost"
	EOM           = "<<< End of mumax message >>>"

	NOTRUNNING  = 0   // CLIENT IS NOT RUNNING
	RUNNING     = 1   // CLIENT IS RUNNING
	TERMINATE   = 255 // CLIENT ASKS MUMAX2 TO TERMINATE ALL THE CLIENTS
	EXIT        = 254 // CLIENT ASKS MUMAX2 TO EXIT, ENSURING THAT ALL OTHER JOBS HAVE BEEN DONE
	LOGFILENAME = "hotspin.log"
)
