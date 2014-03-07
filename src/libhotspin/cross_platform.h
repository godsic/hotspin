#ifndef _CROSSPLATFORM_H
#define _CROSSPLATFORM_H

#if __GNUC__ >= 4
#define DLLEXPORT __attribute__ ((visibility ("default")))
#else
#define DLL_PUBLIC
#endif

#endif
