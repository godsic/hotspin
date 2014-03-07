// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package runtime

// This file provides access to CUDA error statuses.

// CUDA error statuses are not returned by functions but checked and passed to
// panic() when not successful. If desired, they can be caught by
// recover() which will return a cuda.Error.

//#include <cuda_runtime.h>
import "C"
import ()

// CUDA error status
type Error int

// Returns the last error and resets it to cuda.Success
func GetLastError() Error {
	return Error(C.cudaGetLastError())
}

// Returns the last error but does not reset it to cuda.Success
func PeekAtLastError() Error {
	return Error(C.cudaPeekAtLastError())
}

// Message string for the error
func GetErrorString(err Error) string {
	return C.GoString(C.cudaGetErrorString(C.cudaError_t(int(err))))
}

// Message string for the error
func (err Error) String() string {
	return GetErrorString(err)
}

// CUDA error status
const (
	Success                         Error = C.cudaSuccess
	ErrorMissingConfiguration       Error = C.cudaErrorMissingConfiguration
	ErrorMemoryAllocation           Error = C.cudaErrorMemoryAllocation
	ErrorInitializationError        Error = C.cudaErrorInitializationError
	ErrorLaunchFailure              Error = C.cudaErrorLaunchFailure
	ErrorPriorLaunchFailure         Error = C.cudaErrorPriorLaunchFailure
	ErrorLaunchTimeout              Error = C.cudaErrorLaunchTimeout
	ErrorLaunchOutOfResources       Error = C.cudaErrorLaunchOutOfResources
	ErrorInvalidDeviceFunction      Error = C.cudaErrorInvalidDeviceFunction
	ErrorInvalidConfiguration       Error = C.cudaErrorInvalidConfiguration
	ErrorInvalidDevice              Error = C.cudaErrorInvalidDevice
	ErrorInvalidValue               Error = C.cudaErrorInvalidValue
	ErrorInvalidPitchValue          Error = C.cudaErrorInvalidPitchValue
	ErrorInvalidSymbol              Error = C.cudaErrorInvalidSymbol
	ErrorMapBufferObjectFailed      Error = C.cudaErrorMapBufferObjectFailed
	ErrorUnmapBufferObjectFailed    Error = C.cudaErrorUnmapBufferObjectFailed
	ErrorInvalidHostPointer         Error = C.cudaErrorInvalidHostPointer
	ErrorInvalidDevicePointer       Error = C.cudaErrorInvalidDevicePointer
	ErrorInvalidTexture             Error = C.cudaErrorInvalidTexture
	ErrorInvalidTextureBinding      Error = C.cudaErrorInvalidTextureBinding
	ErrorInvalidChannelDescriptor   Error = C.cudaErrorInvalidChannelDescriptor
	ErrorInvalidMemcpyDirection     Error = C.cudaErrorInvalidMemcpyDirection
	ErrorAddressOfConstant          Error = C.cudaErrorAddressOfConstant
	ErrorTextureFetchFailed         Error = C.cudaErrorTextureFetchFailed
	ErrorTextureNotBound            Error = C.cudaErrorTextureNotBound
	ErrorSynchronizationError       Error = C.cudaErrorSynchronizationError
	ErrorInvalidFilterSetting       Error = C.cudaErrorInvalidFilterSetting
	ErrorInvalidNormSetting         Error = C.cudaErrorInvalidNormSetting
	ErrorMixedDeviceExecution       Error = C.cudaErrorMixedDeviceExecution
	ErrorCudartUnloading            Error = C.cudaErrorCudartUnloading
	ErrorUnknown                    Error = C.cudaErrorUnknown
	ErrorNotYetImplemented          Error = C.cudaErrorNotYetImplemented
	ErrorMemoryValueTooLarge        Error = C.cudaErrorMemoryValueTooLarge
	ErrorInvalidResourceHandle      Error = C.cudaErrorInvalidResourceHandle
	ErrorNotReady                   Error = C.cudaErrorNotReady
	ErrorInsufficientDriver         Error = C.cudaErrorInsufficientDriver
	ErrorSetOnActiveProcess         Error = C.cudaErrorSetOnActiveProcess
	ErrorInvalidSurface             Error = C.cudaErrorInvalidSurface
	ErrorNoDevice                   Error = C.cudaErrorNoDevice
	ErrorECCUncorrectable           Error = C.cudaErrorECCUncorrectable
	ErrorSharedObjectSymbolNotFound Error = C.cudaErrorSharedObjectSymbolNotFound
	ErrorSharedObjectInitFailed     Error = C.cudaErrorSharedObjectInitFailed
	ErrorUnsupportedLimit           Error = C.cudaErrorUnsupportedLimit
	ErrorDuplicateVariableName      Error = C.cudaErrorDuplicateVariableName
	ErrorDuplicateTextureName       Error = C.cudaErrorDuplicateTextureName
	ErrorDuplicateSurfaceName       Error = C.cudaErrorDuplicateSurfaceName
	ErrorDevicesUnavailable         Error = C.cudaErrorDevicesUnavailable
	ErrorInvalidKernelImage         Error = C.cudaErrorInvalidKernelImage
	ErrorNoKernelImageForDevice     Error = C.cudaErrorNoKernelImageForDevice
	ErrorIncompatibleDriverContext  Error = C.cudaErrorIncompatibleDriverContext
	ErrorStartupFailure             Error = C.cudaErrorStartupFailure
	ErrorApiFailureBase             Error = C.cudaErrorApiFailureBase
	ErrorHostMemoryAlreadyRegistered    Error = C.cudaErrorHostMemoryAlreadyRegistered
	ErrorHostMemoryNotRegistered    Error = C.cudaErrorHostMemoryNotRegistered
	ErrorOperatingSystem            Error = C.cudaErrorOperatingSystem
)
