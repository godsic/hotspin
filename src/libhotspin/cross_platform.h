#ifndef _CROSSPLATFORM_H
#define _CROSSPLATFORM_H

#if defined _WIN32 || defined __CYGWIN__
#ifdef _USRDLL
#ifdef __GNUC__
#define DLLEXPORT __attribute__ ((dllexport))
#else
#define DLLEXPORT __declspec(dllexport)
#endif
#else
#ifdef __GNUC__
#define DLLEXPORT __attribute__ ((dllimport))
#else
#define DLLEXPORT __declspec(dllimport)
#endif
#endif
#define DLL_LOCAL
#else
#if __GNUC__ >= 4
#define DLLEXPORT __attribute__ ((visibility ("default")))
#else
#define DLL_PUBLIC
#endif
#endif

#endif
