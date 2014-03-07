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
	"strings"
)

// Does not generate a true API library,
// but a TeX file with API documentation.
type Tex struct{}

func (p *Tex) Filename() string {
	return pkg + ".tex"
}

func (p *Tex) Comment() string {
	return "%"
}

func (p *Tex) WriteHeader(out io.Writer) {
	fmt.Fprintln(out, `
`)
}

func (p *Tex) WriteFooter(out io.Writer) {

}

func (p *Tex) WriteFunc(out io.Writer, name string, comment []string, argNames []string, argTypes []reflect.Type, returnTypes []reflect.Type) {

	defer func() {
		err := recover()
		if err != nil {
			fmt.Fprintln(os.Stderr, "WriteFunc ", name, comment, argNames, err)
		}
	}()

	fmt.Fprintln(out)

	fmt.Fprintln(out, `\subsection{`+sanitize(name)+`}`)
	fmt.Fprintln(out, `\label{`+(name)+`}`)
	fmt.Fprintln(out, `\index{`+(sanitize(name))+`}`)

	fmt.Fprint(out, `\texttt{\textbf{`+sanitize(name), `}(`)

	args := ""
	for i := range argTypes {
		if i != 0 {
			args += ", "
		}
		args += argNames[i]
	}
	fmt.Fprintln(out, args+`)}\\`)

	fmt.Fprintln(out, catComment(comment))

	if len(argTypes) > 0 {
		fmt.Fprintln(out, `\textbf{parameter types:}\\`)
		for i := range argTypes {
			fmt.Fprintln(out, argNames[i], `:`, argTypes[i], `\\`)
		}
	}

	if len(returnTypes) > 0 {

		fmt.Fprintln(out, `\textbf{returns:}`)
		for i := range returnTypes {
			fmt.Fprint(out, sanitize(fmt.Sprint(returnTypes[i])), " ")
		}
	}

	//	fmt.Fprintf(out, `	ret = call("%s", [%s])`, name, args)
	//	fmt.Fprint(out, "\n	return ")
	//	for i := range returnTypes {
	//		if i != 0 {
	//			fmt.Fprint(out, ", ")
	//		}
	//		fmt.Fprintf(out, `%v(ret[%v])`, python_convert[returnTypes[i].String()], i)
	//	}
	fmt.Fprintln(out)
	//fmt.Fprintln(out, fmt.Sprintf(`	return %s(call("%s", [%s])[0])`, python_convert[retType], name, args)) // single return value only
}

func sanitize(str string) string {
	const ALL = -1
	str = strings.Replace(str, `_`, `\_`, ALL)
	str = strings.Replace(str, `#`, `\#`, ALL)
	str = strings.Replace(str, `@param`, `\textbf{parameter:}`, ALL)
	str = strings.Replace(str, `@note`, `\textbf{note:}`, ALL)
	str = strings.Replace(str, `[`, `\[`, ALL)
	str = strings.Replace(str, `]`, `\]`, ALL)
	return str
}

func catComment(lines []string) string {
	str := ""
	if len(lines) == 0 {
		return ""
	}
	for _, l := range lines {
		str += sanitize(l) + "\\\\\n"
	}
	return str
}
