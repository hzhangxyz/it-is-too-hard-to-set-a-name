#!/usr/bin/env bash
for Sym in No Z2 U1 Fermi FermiZ2 FermiU1
do
   for ShortScalar in "S float" "D double" "C std::complex<float>" "Z std::complex<double>"
   do
      Short=`awk '{print $1}' <<< $ShortScalar`
      Scalar=`awk '{print $2}' <<< $ShortScalar`
      cat << PYTAT_DEALING > generated_code/dealing_Tensor_$Short$Sym.cpp
#define FUNCTION_NAME dealing_Tensor_${Short}${Sym}
#define SCALAR_NAME ${Scalar}
#define SYMMETRY_NAME ${Sym}Symmetry
#include "../dealing_Tensor.ipp"
PYTAT_DEALING
   done
done
