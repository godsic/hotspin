// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package runtime

import (
	"fmt"
	"runtime"
	"testing"
)

func TestStream(t *testing.T) {
	defer func() {
		err := recover()
		if err != nil {
			t.Error(err)
		}
	}()

	runtime.LockOSThread()

	fmt.Println("StreamCreate")
	str := StreamCreate()
	fmt.Println("StreamQuery:", str.Query())
	fmt.Println("StreamSynchronize")
	str.Synchronize()

	fmt.Println("StreamDestroy")
	(&str).Destroy()
}
