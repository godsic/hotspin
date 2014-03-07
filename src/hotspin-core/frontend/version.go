//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// This file implements version checking
// Author: Arne Vansteenkiste

package frontend

import (
	"io/ioutil"
	"net/http"
	"strconv"
	"strings"
)

// Read a text file on a webserver that should contain a single string representing an integer.
func GetLatestVersionNumber(url string) (version int) {
	// Don't crash. Ever.
	defer func() {
		recover()
	}()

	var client http.Client
	resp, err := client.Get(url)
	if err != nil {
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == 200 { // OK
		bodybuf, err2 := ioutil.ReadAll(resp.Body)
		if err2 != nil {
			return
		}
		body := strings.Trim(string(bodybuf), " \n\t")
		server, err3 := strconv.Atoi(body)
		if err3 != nil {
			return
		}
		version = server
	}
	return
}
