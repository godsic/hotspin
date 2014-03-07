//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// This package implements automated mumax API generation.
// Based on the exported methods of engine.API, an API
// library in any of the supported programming languages is
// automatically generated.
//
// Author: Arne Vansteenkiste
package apigen

import (
	"fmt"
	"io/ioutil"
	. "hotspin-core/common"
	"os"
	"strings"
)

type header struct {
	funcname string
	comment  []string
	args     []string
}

// Auto-generate API libraries for all languages.
func parseSource(srcfile string) map[string]header {
	headers := make(map[string]header)

	// Read api.go
	buf, err := ioutil.ReadFile(srcfile)
	CheckIO(err)
	file := string(buf)

	// 
	lines := strings.Split(file, "\n")
	for i, line := range lines {
		if strings.HasPrefix(line, "func") {
			var head header
			head.comment = []string{}
			funcline := lines[i] // line that starts with func...
			j := i - 1
			for strings.HasPrefix(lines[j], "//") {
				head.comment = append([]string{lines[j][2:]}, head.comment...)
				j--
			}
			head.funcname, head.args = parseFunc(funcline)
			headers[strings.ToLower(head.funcname)] = head
		}
	}
	//fmt.Println(headers)
	return headers
}

func parseFunc(line string) (name string, args []string) {
	defer func() {
		err := recover()
		if err != nil {
			debug("not parsing", line)
			name = ""
			args = nil
		}
	}()

	//func (a API) Name (args) {

	name = line[index(line, ')', 1)+1 : index(line, '(', 2)]
	name = strings.Trim(name, " ")
	argl := line[index(line, '(', 2)+1 : index(line, ')', 2)]
	args = parseArgs(argl)
	return
}

func parseArgs(line string) (args []string) {
	args = strings.Split(line, ",")
	for i := range args {
		args[i] = strings.Trim(args[i], " ")
		words := strings.Split(args[i], " ")
		args[i] = strings.Trim(words[0], " ") // remove type, if any
	}
	return
}

func debug(msg ...interface{}) {
	fmt.Fprintln(os.Stderr, msg...)
}

// index of nth occurrence of sep in s.
func index(s string, sep uint8, n int) int {
	for i := range s {
		if s[i] == sep {
			n--
		}
		if n == 0 {
			return i
		}
	}
	return -1
}
