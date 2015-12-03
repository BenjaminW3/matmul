//-----------------------------------------------------------------------------
//! \file
//! Copyright 2015 Benjamin Worpitz
//!
//! This file is part of matmul.
//!
//! matmul is free software: you can redistribute it and/or modify
//! it under the terms of the GNU Lesser General Public License as published by
//! the Free Software Foundation, either version 3 of the License, or
//! (at your option) any later version.
//!
//! matmul is distributed in the hope that it will be useful,
//! but WITHOUT ANY WARRANTY; without even the implied warranty of
//! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//! GNU Lesser General Public License for more details.
//!
//! You should have received a copy of the GNU Lesser General Public License
//! along with matmul.
//! If not, see <http://www.gnu.org/licenses/>.
//-----------------------------------------------------------------------------

#include <matmul/common/Time.h>

#include <assert.h>                 // assert
#include <time.h>                   // time()

#ifdef _MSC_VER
    #include <Windows.h>
    //-----------------------------------------------------------------------------
    //! \return A monotonically increasing time value in seconds.
    //-----------------------------------------------------------------------------
    double getTimeSec()
    {
        LARGE_INTEGER li, frequency;
        BOOL bSucQueryPerformanceCounter = QueryPerformanceCounter(&li);
        if(bSucQueryPerformanceCounter)
        {
            BOOL bSucQueryPerformanceFrequency = QueryPerformanceFrequency(&frequency);
            if(bSucQueryPerformanceFrequency)
            {
                return ((double)li.QuadPart)/((double)frequency.QuadPart);
            }
            else
            {
                // Throw assertion in debug mode, else return 0 time.
                assert(bSucQueryPerformanceFrequency);
                return 0.0;
            }
        }
        else
        {
            // Throw assertion in debug mode, else return 0 time.
            assert(bSucQueryPerformanceCounter);
            return 0.0;
        }
    }
#else
    #include <sys/time.h>
    //-----------------------------------------------------------------------------
    //! \return A monotonically increasing time value in seconds.
    //-----------------------------------------------------------------------------
    double getTimeSec()
    {
        struct timeval act_time;
        gettimeofday(&act_time, NULL);
        return (double)act_time.tv_sec + (double)act_time.tv_usec / 1000000.0;
    }
#endif
