
// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
*                                                                    *
* Copyright (c) 2005 , LCG ROOT FNAL MathLib Team                    *
*                                                                    *
*                                                                    *
**********************************************************************/

// Header file for class PtEtaPhiE4D
//
// Created by: fischler at Wed Jul 20 2005
//   based on CylindricalEta4D by moneta
//
// Last update: $Id$
//
#ifndef ROOT_Experimental_GenVector_PtEtaPhiE4D
#define ROOT_Experimental_GenVector_PtEtaPhiE4D  1

#undef __MAKECINT__

#include "SYCLMath/GenVector/etaMax.h"
#include "SYCLMath/GenVector/MathUtil.h"

//#define TRACE_CE
#ifdef TRACE_CE
#include <iostream>
#endif


namespace ROOT {

namespace Experimental {

//__________________________________________________________________________________________
/**
    Class describing a 4D cylindrical coordinate system
    using Pt , Phi, Eta and E (or rho, phi, eta , T)
    The metric used is (-,-,-,+).
    Phi is restricted to be in the range [-PI,PI)

    @ingroup GenVector

    @sa Overview of the @ref GenVector "physics vector library"
*/

template <class ScalarType>
class PtEtaPhiE4D {

public :

   typedef ScalarType Scalar;

   // --------- Constructors ---------------

   /**
      Default constructor gives zero 4-vector
   */
   __roohost__ __roodevice__ PtEtaPhiE4D () : fPt(0), fEta(0), fPhi(0), fE(0) { }

   /**
      Constructor  from pt, eta, phi, e values
   */
   __roohost__ __roodevice__ PtEtaPhiE4D (Scalar pt, Scalar eta, Scalar phi, Scalar e) :
      fPt(pt), fEta(eta), fPhi(phi), fE(e) { Restrict(); }

   /**
      Generic constructor from any 4D coordinate system implementing
      Pt(), Eta(), Phi() and E()
   */
   template <class CoordSystem >
    __roohost__ __roodevice__ explicit PtEtaPhiE4D (const CoordSystem & c) :
      fPt(c.Pt()), fEta(c.Eta()), fPhi(c.Phi()), fE(c.E())  { }

   // for g++  3.2 and 3.4 on 32 bits found that the compiler generated copy ctor and assignment are much slower
   // so we decided to re-implement them ( there is no no need to have them with g++4)

   /**
      copy constructor
    */
   __roohost__ __roodevice__ PtEtaPhiE4D (const PtEtaPhiE4D & v) :
      fPt(v.fPt), fEta(v.fEta), fPhi(v.fPhi), fE(v.fE) { }

   /**
      assignment operator
    */
   __roohost__ __roodevice__ PtEtaPhiE4D  & operator = (const PtEtaPhiE4D & v) {
      fPt  = v.fPt;
      fEta = v.fEta;
      fPhi = v.fPhi;
      fE   = v.fE;
      return *this;
   }


   /**
      Set internal data based on an array of 4 Scalar numbers
   */
   __roohost__ __roodevice__  void  SetCoordinates( const Scalar src[] )
   { fPt=src[0]; fEta=src[1]; fPhi=src[2]; fE=src[3]; Restrict(); }

   /**
      get internal data into an array of 4 Scalar numbers
   */
   __roohost__ __roodevice__  void  GetCoordinates( Scalar dest[] ) const
   { dest[0] = fPt; dest[1] = fEta; dest[2] = fPhi; dest[3] = fE; }

   /**
      Set internal data based on 4 Scalar numbers
   */
   __roohost__ __roodevice__  void  SetCoordinates(Scalar pt, Scalar eta, Scalar phi, Scalar e)
   { fPt=pt; fEta = eta; fPhi = phi; fE = e; Restrict(); }

   /**
      get internal data into 4 Scalar numbers
   */
   __roohost__ __roodevice__  void 
   GetCoordinates(Scalar& pt, Scalar & eta, Scalar & phi, Scalar& e) const
   { pt=fPt; eta=fEta; phi = fPhi; e = fE; }

   // --------- Coordinates and Coordinate-like Scalar properties -------------

   // 4-D Cylindrical eta coordinate accessors

   __roohost__ __roodevice__  Scalar  Pt()  const { return fPt;  }
   __roohost__ __roodevice__  Scalar  Eta() const { return fEta; }
   __roohost__ __roodevice__  Scalar  Phi() const { return fPhi; }
   __roohost__ __roodevice__  Scalar  E()   const { return fE;   }

   __roohost__ __roodevice__  Scalar  Perp()const { return Pt(); }
   __roohost__ __roodevice__  Scalar  Rho() const { return Pt(); }
   __roohost__ __roodevice__  Scalar  T()   const { return E();  }

   // other coordinate representation

   __roohost__ __roodevice__  Scalar  Px() const { return fPt * mycos(fPhi); }
   __roohost__ __roodevice__  Scalar  X () const { return Px();         }
   __roohost__ __roodevice__  Scalar  Py() const { return fPt * mysin(fPhi); }
   __roohost__ __roodevice__  Scalar  Y () const { return Py();         }
   __roohost__ __roodevice__  Scalar  Pz() const {
      return fPt > 0 ? fPt * mysinh(fEta) : fEta == 0 ? 0 : fEta > 0 ? fEta - etaMax<Scalar>() : fEta + etaMax<Scalar>();
   }
   __roohost__ __roodevice__  Scalar  Z () const { return Pz(); }

   /**
       magnitude of momentum
   */
   __roohost__ __roodevice__  Scalar  P() const {
      return fPt > 0 ? fPt * mycosh(fEta)
                     : fEta > etaMax<Scalar>() ? fEta - etaMax<Scalar>()
                                               : fEta < -etaMax<Scalar>() ? -fEta - etaMax<Scalar>() : 0;
   }
   __roohost__ __roodevice__  Scalar  R() const { return P(); }

   /**
       squared magnitude of spatial components (momentum squared)
   */
   __roohost__ __roodevice__  Scalar  P2() const
   {
      const Scalar p = P();
      return p * p;
   }

   /**
      vector magnitude squared (or mass squared)
   */
   __roohost__ __roodevice__  Scalar  M2() const
   {
      const Scalar p = P();
      return fE * fE - p * p;
   }
   __roohost__ __roodevice__  Scalar  Mag2() const { return M2(); }

   /**
      invariant mass
   */
   __roohost__ __roodevice__  Scalar  M() const    {
      const Scalar mm = M2();
      if (mm >= 0) {
         return mysqrt(mm);
      } else {
         //GenVector::Throw ("PtEtaPhiE4D::M() - Tachyonic:\n"
         //                  "    Pt and Eta give P such that P^2 > E^2, so the mass would be imaginary");
         return -mysqrt(-mm);
      }
   }
   __roohost__ __roodevice__  Scalar  Mag() const    { return M(); }

   /**
       transverse spatial component squared
   */
   __roohost__ __roodevice__ Scalar Pt2()   const { return fPt*fPt;}
   __roohost__ __roodevice__ Scalar Perp2() const { return Pt2();  }

   /**
       transverse mass squared
   */
   __roohost__ __roodevice__ Scalar Mt2() const {  Scalar pz = Pz(); return fE*fE  - pz*pz; }

   /**
      transverse mass
   */
   __roohost__ __roodevice__ Scalar Mt() const {
      const Scalar mm = Mt2();
      if (mm >= 0) {
         return mysqrt(mm);
      } else {
         //GenVector::Throw ("PtEtaPhiE4D::Mt() - Tachyonic:\n"
         //                  "    Pt and Eta give Pz such that Pz^2 > E^2, so the mass would be imaginary");
         return -mysqrt(-mm);
      }
   }

   /**
      transverse energy
   */
   /**
      transverse energy
   */
   __roohost__ __roodevice__ Scalar Et() const {

      return fE / mycosh(fEta); // faster using eta
   }

   /**
       transverse energy squared
   */
   __roohost__ __roodevice__ Scalar Et2() const
   {
      const Scalar et = Et();
      return et * et;
   }

private:
    __roohost__ __roodevice__ static Scalar pi() { return M_PI; }
    __roohost__ __roodevice__  void  Restrict() {
      if (fPhi <= -pi() || fPhi > pi()) fPhi = fPhi - myfloor(fPhi / (2 * pi()) + .5) * 2 * pi();
   }
public:

   /**
      polar angle
   */
   __roohost__ __roodevice__ Scalar Theta() const {  return (fPt > 0 ? Scalar(2) * myatan(myexp(-fEta)) : fEta >= 0 ? 0 : pi()); }

   // --------- Set Coordinates of this system  ---------------

   /**
      set Pt value
   */
   __roohost__ __roodevice__  void  SetPt( Scalar  pt) {
      fPt = pt;
   }
   /**
      set eta value
   */
   __roohost__ __roodevice__  void  SetEta( Scalar  eta) {
      fEta = eta;
   }
   /**
      set phi value
   */
   __roohost__ __roodevice__  void  SetPhi( Scalar  phi) {
      fPhi = phi;
      Restrict();
   }
   /**
      set E value
   */
   __roohost__ __roodevice__  void  SetE( Scalar  e) {
      fE = e;
   }

   /**
       set values using cartesian coordinate system
   */
   __roohost__ __roodevice__  void  SetPxPyPzE(Scalar px, Scalar py, Scalar pz, Scalar e);


   // ------ Manipulations -------------

   /**
      negate the 4-vector
   */
   __roohost__ __roodevice__  void  Negate( ) {
      fPhi = ( fPhi > 0 ? fPhi - pi() : fPhi + pi()  );
      fEta = - fEta;
      fE = - fE;
   }

   /**
      Scale coordinate values by a scalar quantity a
   */
   __roohost__ __roodevice__  void  Scale( Scalar a) {
      if (a < 0) {
         Negate(); a = -a;
      }
      fPt *= a;
      fE  *= a;
   }

   /**
      Assignment from a generic coordinate system implementing
      Pt(), Eta(), Phi() and E()
   */
   template <class CoordSystem >
   __roohost__ __roodevice__ PtEtaPhiE4D  & operator = (const CoordSystem & c) {
      fPt  = c.Pt();
      fEta = c.Eta();
      fPhi = c.Phi();
      fE   = c.E();
      return *this;
   }

   /**
      Exact equality
   */
   __roohost__ __roodevice__ bool operator == (const PtEtaPhiE4D & rhs) const {
      return fPt == rhs.fPt && fEta == rhs.fEta
         && fPhi == rhs.fPhi && fE == rhs.fE;
   }
   __roohost__ __roodevice__ bool operator != (const PtEtaPhiE4D & rhs) const {return !(operator==(rhs));}

   // ============= Compatibility section ==================

   // The following make this coordinate system look enough like a CLHEP
   // vector that an assignment member template can work with either
   __roohost__ __roodevice__ Scalar x() const { return X(); }
   __roohost__ __roodevice__ Scalar y() const { return Y(); }
   __roohost__ __roodevice__ Scalar z() const { return Z(); }
   __roohost__ __roodevice__ Scalar t() const { return E(); }



#if defined(__MAKECINT__) || defined(G__DICTIONARY)

   // ====== Set member functions for coordinates in other systems =======

   __roohost__ __roodevice__  void  SetPx(Scalar px);

   __roohost__ __roodevice__  void  SetPy(Scalar py);

   __roohost__ __roodevice__  void  SetPz(Scalar pz);

   __roohost__ __roodevice__  void  SetM(Scalar m);


#endif

private:

   ScalarType fPt;
   ScalarType fEta;
   ScalarType fPhi;
   ScalarType fE;

};


} // end namespace Experimental
} // end namespace ROOT



// move implementations here to avoid circle dependencies
#include "SYCLMath/GenVector/PxPyPzE4D.h"
#if defined(__MAKECINT__) || defined(G__DICTIONARY)
#include "SYCLMath/GenVector/PtEtaPhiM4D.h"
#include "SYCLMath/GenVector/GenVector_exception.h"
#endif

namespace ROOT {

namespace Experimental {

template <class ScalarType>
 __roohost__ __roodevice__  void  PtEtaPhiE4D<ScalarType>::SetPxPyPzE(Scalar px, Scalar py, Scalar pz, Scalar e) {
   *this = PxPyPzE4D<Scalar> (px, py, pz, e);
}


#if defined(__MAKECINT__) || defined(G__DICTIONARY)

  // ====== Set member functions for coordinates in other systems =======

template <class ScalarType>
 __roohost__ __roodevice__  void  PtEtaPhiE4D<ScalarType>::SetPx(Scalar px) {
   GenVector_exception e("PtEtaPhiE4D::SetPx() is not supposed to be called");
   throw e;
   PxPyPzE4D<Scalar> v(*this); v.SetPx(px); *this = PtEtaPhiE4D<Scalar>(v);
}
template <class ScalarType>
 __roohost__ __roodevice__  void  PtEtaPhiE4D<ScalarType>::SetPy(Scalar py) {
   GenVector_exception e("PtEtaPhiE4D::SetPx() is not supposed to be called");
   throw e;
   PxPyPzE4D<Scalar> v(*this); v.SetPy(py); *this = PtEtaPhiE4D<Scalar>(v);
}
template <class ScalarType>
 __roohost__ __roodevice__  void  PtEtaPhiE4D<ScalarType>::SetPz(Scalar pz) {
   GenVector_exception e("PtEtaPhiE4D::SetPx() is not supposed to be called");
   throw e;
   PxPyPzE4D<Scalar> v(*this); v.SetPz(pz); *this = PtEtaPhiE4D<Scalar>(v);
}
template <class ScalarType>
 __roohost__ __roodevice__  void  PtEtaPhiE4D<ScalarType>::SetM(Scalar m) {
   GenVector_exception e("PtEtaPhiE4D::SetM() is not supposed to be called");
   throw e;
   PtEtaPhiM4D<Scalar> v(*this); v.SetM(m);
   *this = PtEtaPhiE4D<Scalar>(v);
}

#endif  // endif __MAKE__CINT || G__DICTIONARY

} // end namespace Experimental

} // end namespace ROOT




#endif // ROOT_Experimental_GenVector_PtEtaPhiE4D
