/*
Copyright (c) 2014-2015, Scott Zuyderduyn
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
*/

//----------------------------------------------------------------------------
// event.hpp
//
// Encapsulates a CUDA API event.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_EVENT_HPP
#define ECUDA_EVENT_HPP

#include "impl/host_emulation.hpp" // gets data structure definitions when compiling host-only without nvcc

namespace ecuda {

///
/// \brief Encapsulates CUDA API event objects and functions.
///
/// CUDA events are useful for assessing the running time of device operations.
/// This can be handy when optimizing kernel implementations or thread
/// configurations.
///
/// This is just a thin wrapper around the appropriate cudaEventXXXXX functions
/// in the CUDA API to provide access to the event functions in a more C++-like
/// style.  The documentation is shamelessly lifted from the official CUDA
/// documentation (http://docs.nvidia.com/cuda/index.html).
///
/// For example, to get the running time of a kernel function:
///
/// \code{.cpp}
/// ecuda::event start, stop;
///
/// // ... specify thread grid/blocks
///
/// start.record();
/// kernelFunction<<<grid,block>>>( ... ); // call the kernel
/// stop.record();
/// stop.synchronize(); // wait until kernel finishes executing
///
/// std::cout << "EXECUTION TIME: " << ( stop - start ) << "ms" << std::endl;
/// \endcode
///
class event {

private:
	cudaEvent_t _event;

public:
	///
	/// \brief Default constructor.
	///
	/// Creates a default event object.
	///
	event() { cudaEventCreate(&_event); }

	///
	/// \brief Constructs an event with the given flags.
	///
	/// As of now, valid flags specified by the CUDA API are:
	///   - cudaEventDefault: Default event creation flag.
	///   - cudaEventBlockingSync: Specifies that event should use blocking synchronization.\n
	///     A host thread that uses synchronize() to wait on an event created with this flag will block until the event actually completes.
	///   - cudaEventDisableTiming: Specifies that the created event does not need to record timing data.\n
	///     Events created with this flag specified and the cudaEventBlockingSync flag not specified will provide the best performance when used
	///     with cudaStreamWaitEvent() and query().
	///
	/// \param flags A valid event flag.
	///
	event( unsigned flags ) { cudaEventCreateWithFlags(&_event,flags); }

	///
	/// \brief Destructor.
	///
	/// Deallocates the underlying CUDA event and destroys this object.
	///
	~event() { cudaEventDestroy(_event); }

	///
	/// \brief Records an event.
	///
	/// If stream is non-zero, the event is recorded after all preceding operations in stream have been completed;
	/// otherwise, it is recorded after all preceding operations in the CUDA context have been completed. Since
	/// operation is asynchronous, query() and/or synchronize() must be used to determine when the event has actually
	/// been recorded.
	///
	/// If record() has previously been called on event, then this call will overwrite any existing state
	/// in event. Any subsequent calls which examine the status of event will only examine the completion of this
	/// most recent call to record().
	///
	/// \param stream Stream in which to record event.
	///
	inline void record( cudaStream_t stream = 0 ) { cudaEventRecord( _event, stream ); }

	///
	/// \brief Wait until the completion of all device work preceding the most recent call to record().
	///
	/// This applies to the appropriate compute streams, as specified by the arguments to record().
	/// If record() has not been called on event, cudaSuccess is returned immediately.
	///
	///	Waiting for an event that was created with the cudaEventBlockingSync flag will cause the calling
	/// CPU thread to block until the event has been completed by the device. If the cudaEventBlockingSync
	/// flag has not been set, then the CPU thread will busy-wait until the event has been completed by the
	/// device.
	///
	/// \returns cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle,
	///          cudaErrorLaunchFailure
	///
	inline cudaError_t synchronize() { return cudaEventSynchronize( _event ); }

	///
	/// \brief Query the status of all device work preceding the most recent call to record().
	///
	/// This applies to the appropriate compute streams, as specified by the arguments to record().
	///
	/// If this work has successfully been completed by the device, or if record() has not been called on event,
	/// then cudaSuccess is returned. If this work has not yet been completed by the device then cudaErrorNotReady
	/// is returned.
	///
	/// \return cudaSuccess, cudaErrorNotReady, cudaErrorInitializationError, cudaErrorInvalidValue,
	///         cudaErrorInvalidResourceHandle, cudaErrorLaunchFailure
	///
	inline cudaError_t query() { return cudaEventQuery( _event ); }

	///
	/// \brief Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds).
	///
	/// If either event was last recorded in a non-NULL stream, the resulting time may be greater than expected (even
	/// if both used the same stream handle). This happens because the record() operation takes place asynchronously
	/// and there is no guarantee that the measured latency is actually just between the two events. Any number of
	/// other different stream operations could execute in between the two measured events, thus altering the timing
	/// in a significant way.
	///
	/// If record() has not been called on either event, then cudaErrorInvalidResourceHandle is returned. If record()
	/// has been called on both events but one or both of them has not yet been completed (that is, query() would return
	/// cudaErrorNotReady on at least one of the events), cudaErrorNotReady is returned. If either event was created
	/// with the cudaEventDisableTiming flag, then this function will return cudaErrorInvalidResourceHandle.
	///
	/// \param start Starting event.
	/// \param end Ending event.
	/// \return The elapsed time between the two events in milliseconds.
	///
	static float elapsed_time( event& start, event& end ) {
		float t;
		cudaEventElapsedTime( &t, start._event, end._event );
		return t;
	}

	///
	/// \brief Computes the elapsed time between another event and this event.
	///
	/// This is equivalent to elapsed_time( other, *this ).
	///
	/// \param other The other event with which to compute the elapsed time.
	/// \return The elapsed time between the two events in milliseconds.
	///
	inline float operator-( event& other ) { return event::elapsed_time( other, *this ); }

};


} // namespace ecuda

#endif
