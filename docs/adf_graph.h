
//===----------------------------------------------------------------------===//
//
// Automatically generated file for adf_graph.h
//
//===----------------------------------------------------------------------===//
#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <adf.h>
#include <stdio.h>
#include <iostream>
#include "adf_kernel.h"
using namespace adf;


class spmv_rowblk: public adf::graph{
private:
   ;

public:
  int32_t v0;
  float v1;
  float v2;
  float v3;
  int v4;

  spmv_rowblk() {
    for (int v5 = 0; v5 < v4; v5++) {	// L26
      int v6 = (v5 * 608);	// L31
      for (unsigned iv0=0; iv0<-9223372036854775808; iv0++){
        [iv0] = v0[v6 + iv0 * 1];
      }
      int v7 = (v5 * 608);	// L35
      for (unsigned iv0=0; iv0<-9223372036854775808; iv0++){
        [iv0] = v1[v7 + iv0 * 1];
      }
      int v8 = 0;	// L39
      for (unsigned iv0=0; iv0<-9223372036854775808; iv0++){
        [iv0] = v2[v8 + iv0 * 1];
      }
      kernel_spmv_rowblk(int32_t v9, float v10, float v11, float v12);	// L43
      int v13 = (v5 * 32);	// L44
      for (unsigned iv0=0; iv0<-9223372036854775808; iv0++){
        v3[v13 + iv0 * 1] = v12[iv0];
      }
    }
  }
};

#endif //__GRAPH_H__

