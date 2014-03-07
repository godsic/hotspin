/** 
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *  Note that you are welcome to modify this code under the condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 *
 * @file libmumax2.h
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */

#ifndef _LIBMUMAX2_H
#define _LIBMUMAX2_H

#ifdef __cplusplus
extern "C" {
#endif

#include "energy_flow.h"
#include "llbar-nonlocal00nc.h"
#include "brillouin.h"
#include "reduce.h"
#include "Qinter.h"
#include "long_field.h"
#include "llbar-local02nc.h"
#include "llbar-local02c.h"
#include "add.h"
#include "Ts.h"
#include "Cp.h"
#include "llbar-local00nc.h"
#include "mul.h"
#include "wavg.h"
#include "exchange6.h"
#include "temperature.h"
#include "uniaxialanisotropy.h"
#include "copypad.h"
#include "divMulPow.h"
#include "dot.h"
#include "llbar-torque.h"
#include "kappa.h"
#include "div.h"
#include "decompose.h"
#include "Qspat.h"
#include "normalize.h"


#ifdef __cplusplus
}
#endif
#endif
