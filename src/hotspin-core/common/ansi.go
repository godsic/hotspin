//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

import ()

// ANSI escape sequences
const (
	ESC        = "\033["
	BOLD       = "\033[1m"
	BLACK      = "\033[30m"
	RED        = "\033[31m"
	GREEN      = "\033[32m"
	YELLOW     = "\033[33m"
	BLUE       = "\033[34m"
	MAGENTA    = "\033[35m"
	CYAN       = "\033[36m"
	WHITE      = "\033[37m"
	RESET      = "\033[0m" // No formatting
	ERASE      = "\033[K"  // Erase rest of line
	LINEUP     = "\033[1A"
	LINEDOWN   = "\033[1B"
	LINEBEGIN  = "\033[?0E"
	HIDECURSOR = "\033[?25l"
	SHOWCURSOR = "\033[?25h"
)
