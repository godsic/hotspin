//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package apigen

import (
	"fmt"
	"io"
	"os"
	"reflect"
)

type Python struct{}

func (p *Python) Filename() string {
	return pkg + ".py"
}

func (p *Python) Comment() string {
	return "#"
}

func (p *Python) WriteHeader(out io.Writer) {
	fmt.Fprintln(out, `
import os
import json
import sys
import socket


m_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
initialized = 0
outputdir = ""
M_ADDR = ""
M_HOST, M_PORT = "", 0

## Initializes the communication with mumax2.
# @note Internal use only
def init():
	global outputdir
	global m_sock
	global initialized
	global M_HOST
	global M_PORT
	
	# get the output directory from environment
	outputdir=os.environ["HOTSPIN_OUTPUTDIR"] + "/"	
	M_ADDR = os.environ["HOTSPIN_ADDR"]
	
	print 'MuMax grants connection on: ' + M_ADDR
	
	m_name = M_ADDR.split(':')
	M_HOST = m_name[0]
	M_PORT = int(m_name[1])
	
	m_sock.connect((M_HOST,M_PORT))
	initialized = 1
		
#	print 'python frontend is initialized'

## Calls a mumax2 command and returns the result as string.
# @note Internal use only.
def call(command, args):
	if (initialized == 0):
		init()
	m_sock.sendall(json.dumps([command, args])+'\n')
	resp = recvall(m_sock)	     
	return json.loads(resp)

End='<<< End of mumax message >>>'

## Retrieving message until the EOM statement
# @note Internal use only.
def recvall(the_socket):
    total_data=[];data=''
    while True:
            data=the_socket.recv(8192)
            if data == '':
                sys.exit(1)
            if End in data:
                total_data.append(data[:data.find(End)])
                break
            total_data.append(data)
            if len(total_data)>1:
                #check if end_of_data was split
                last_pair=total_data[-2]+total_data[-1]
                if End in last_pair:
                    total_data[-2]=last_pair[:last_pair.find(End)]
                    total_data.pop()
                    break
    return ''.join(total_data)
		
`)
}

func (p *Python) WriteFooter(out io.Writer) {

}

func (p *Python) WriteFunc(out io.Writer, name string, comment []string, argNames []string, argTypes []reflect.Type, returnTypes []reflect.Type) {
	defer func() {
		err := recover()
		if err != nil {
			fmt.Fprintln(os.Stderr, "WriteFunc ", name, comment, argNames, err)
		}
	}()

	fmt.Fprintln(out)
	fmt.Fprintf(out, pyDocComment(comment))
	fmt.Fprint(out, "def ", name, "(")

	args := ""
	for i := range argTypes {
		if i != 0 {
			args += ", "
		}
		args += argNames[i]
	}
	fmt.Fprintln(out, args, "):")

	fmt.Fprintf(out, `	ret = call("%s", [%s])`, name, args)
	fmt.Fprint(out, "\n	return ")
	for i := range returnTypes {
		if i != 0 {
			fmt.Fprint(out, ", ")
		}
		fmt.Fprintf(out, `%v(ret[%v])`, python_convert[returnTypes[i].String()], i)
	}
	fmt.Fprintln(out)
	//fmt.Fprintln(out, fmt.Sprintf(`	return %s(call("%s", [%s])[0])`, python_convert[retType], name, args)) // single return value only
}

var (
	// maps go types to python types
	python_convert map[string]string = map[string]string{"int": "int",
		"float64": "float",
		"string":  "str",
		"bool":    "bool",
		"":        ""}
)

// Puts python doc comment tokens in front of the comment lines.
func pyDocComment(lines []string) string {
	if len(lines) == 0 {
		return ""
	}
	str := "#"
	for _, l := range lines {
		str += "# " + l + "\n"
	}
	return str
}
