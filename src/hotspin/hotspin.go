//  This file is part of hotspin, a high-performance finite-temperature micromagnetic simulator.
//  Copyright 2014 Mykola Dvornik
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package main

import (
	"hotspin-core/frontend"
	_ "hotspin-core/modules" // register and link core modules
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

const (
	PYLIB            = "PYTHONPATH"
	PYTHONMODULEPATH = "/../python"
	BINNAME          = "hotspin"
)

func main() {

	hotspinBinDir, err := exec.LookPath(BINNAME)
	if err != nil {
		log.Fatal(BINNAME + " is not in the $PATH variable")
	}

	hotspinBinDir = filepath.Dir(hotspinBinDir)

	envValueSep := ":"
	if runtime.GOOS == "windows" {
		envValueSep = ";"
	}

	pyLibValue := os.Getenv(PYLIB)
	pyLibValue += (envValueSep + hotspinBinDir + PYTHONMODULEPATH)
	pyLibValue = filepath.FromSlash(pyLibValue)
	os.Setenv(PYLIB, pyLibValue)

	frontend.Main()
}
