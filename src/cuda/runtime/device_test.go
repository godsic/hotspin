// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package runtime

import (
	"fmt"
	"runtime"
	"testing"
)

func TestDevice(t *testing.T) {
	defer func() {
		err := recover()
		if err != nil {
			t.Error(err)
		}
	}()

	runtime.LockOSThread()
	count := GetDeviceCount()
	for i := 0; i < count; i++ {
		fmt.Println(GetDeviceProperties(i))
		fmt.Println()
	}
	SetDevice(count - 1)
	if GetDevice() != count-1 {
		t.Fail()
	}
	SetDeviceFlags(DeviceScheduleAuto)
}
