package dump

import (
	"fmt"
)

// Magic number
const MAGIC = "#dump002"

// Precision identifier
const (
	FLOAT32 = 4
)

// Header+data frame.
type Frame struct {
	Header
	Data  []float64
	CRC   uint64
	Bytes int64 // Total number of bytes read.
	Err   error // Stores the latest I/O error, if any.
}

// Header for dump data frame
type Header struct {
	Magic      string
	Components int
	MeshSize   [3]int
	MeshStep   [3]float64
	MeshUnit   string
	Time       float64
	TimeUnit   string
	DataLabel  string
	DataUnit   string
	Precission uint64
}

func (h *Header) Size() []int {
	return []int{h.Components, h.MeshSize[0], h.MeshSize[1], h.MeshSize[2]}
}

func (h *Header) String() string {
	return fmt.Sprintf(
		`     Magic: %v
Components: %v
  MeshSize: %v
  MeshStep: %v
  MeshUnit: %v
      Time: %v
  TimeUnit: %v
 DataLabel: %v
  DataUnit: %v
Precission: %v
`, h.Magic, h.Components, h.MeshSize, h.MeshStep, h.MeshUnit, h.Time, h.TimeUnit, h.DataLabel, h.DataUnit, h.Precission)
}
