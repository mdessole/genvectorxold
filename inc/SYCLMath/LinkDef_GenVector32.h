// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

// Linkdef for Doublr32_t types


#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;


#pragma link C++ class    ROOT::Experimental::Cartesian2D<Double32_t>+;
#pragma read sourceClass="ROOT::Experimental::Cartesian2D<double>"    \
             targetClass="ROOT::Experimental::Cartesian2D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::Cartesian2D<float>"     \
             targetClass="ROOT::Experimental::Cartesian2D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::Cartesian2D<Float16_t>" \
             targetClass="ROOT::Experimental::Cartesian2D<Double32_t>";

#pragma link C++ class    ROOT::Experimental::Polar2D<Double32_t>+;
#pragma read sourceClass="ROOT::Experimental::Polar2D<double>"    \
             targetClass="ROOT::Experimental::Polar2D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::Polar2D<float>"     \
             targetClass="ROOT::Experimental::Polar2D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::Polar2D<Float16_t>" \
             targetClass="ROOT::Experimental::Polar2D<Double32_t>";


#pragma link C++ class    ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<Double32_t> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<double> >"    \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<float> >"     \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<Float16_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<Double32_t> >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<Double32_t> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<double> >"    \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<float> >"     \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<Float16_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<Double32_t> >";



#pragma link C++ class    ROOT::Experimental::Cartesian3D<Double32_t>+;
#pragma read sourceClass="ROOT::Experimental::Cartesian3D<double>"    \
             targetClass="ROOT::Experimental::Cartesian3D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::Cartesian3D<float>"     \
             targetClass="ROOT::Experimental::Cartesian3D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::Cartesian3D<Float16_t>" \
             targetClass="ROOT::Experimental::Cartesian3D<Double32_t>";

#pragma link C++ class    ROOT::Experimental::CylindricalEta3D<Double32_t>+;
#pragma read sourceClass="ROOT::Experimental::CylindricalEta3D<double>"    \
             targetClass="ROOT::Experimental::CylindricalEta3D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::CylindricalEta3D<float>"     \
             targetClass="ROOT::Experimental::CylindricalEta3D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::CylindricalEta3D<Float16_t>" \
             targetClass="ROOT::Experimental::CylindricalEta3D<Double32_t>";

#pragma link C++ class    ROOT::Experimental::Polar3D<Double32_t>+;
#pragma read sourceClass="ROOT::Experimental::Polar3D<double>"    \
             targetClass="ROOT::Experimental::Polar3D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::Polar3D<float>"     \
             targetClass="ROOT::Experimental::Polar3D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::Polar3D<Float16_t>" \
             targetClass="ROOT::Experimental::Polar3D<Double32_t>";

#pragma link C++ class    ROOT::Experimental::Cylindrical3D<Double32_t>+;
#pragma read sourceClass="ROOT::Experimental::Cylindrical3D<double>"    \
             targetClass="ROOT::Experimental::Cylindrical3D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::Cylindrical3D<float>"     \
             targetClass="ROOT::Experimental::Cylindrical3D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::Cylindrical3D<Float16_t>" \
             targetClass="ROOT::Experimental::Cylindrical3D<Double32_t>";



#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<double> >"    \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<float> >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Float16_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t> >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<double> >"    \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<float> >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Float16_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t> >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<Double32_t> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<double> >"    \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<float> >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<Float16_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<Double32_t> >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<Double32_t> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<double> >"    \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<float> >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<Float16_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<Double32_t> >";


#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t> >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<double> >"    \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<float> >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Float16_t> >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t> >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t> >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<double> >"    \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<float> >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Float16_t> >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t> >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<Double32_t> >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<double> >"    \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<float> >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<Float16_t> >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<Double32_t> >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<Double32_t> >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<double> >"    \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<float> >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<Float16_t> >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<Double32_t> >";


// using a tag (only cartesian and cylindrical eta)

#ifdef __CLING__
// Work around CINT and autoloader deficiency with template default parameter
// Those requests as solely for rlibmap, they do no need to be seen by rootcint
#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >"    \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >"    \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >";


#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >"    \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >"    \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >";

#endif

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t>, ROOT::Experimental::LocalCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<double>, ROOT::Experimental::LocalCoordinateSystemTag >"    \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t>, ROOT::Experimental::LocalCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<float>, ROOT::Experimental::LocalCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t>, ROOT::Experimental::LocalCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Float16_t>, ROOT::Experimental::LocalCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t>, ROOT::Experimental::LocalCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::LocalCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<double>,ROOT::Experimental::LocalCoordinateSystemTag >"    \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::LocalCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<float>,ROOT::Experimental::LocalCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::LocalCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Float16_t>,ROOT::Experimental::LocalCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::LocalCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t>, ROOT::Experimental::GlobalCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<double>, ROOT::Experimental::GlobalCoordinateSystemTag >"    \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t>, ROOT::Experimental::GlobalCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<float>, ROOT::Experimental::GlobalCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t>, ROOT::Experimental::GlobalCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Float16_t>, ROOT::Experimental::GlobalCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t>, ROOT::Experimental::GlobalCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::GlobalCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<double>,ROOT::Experimental::GlobalCoordinateSystemTag >"    \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::GlobalCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<float>,ROOT::Experimental::GlobalCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::GlobalCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Float16_t>,ROOT::Experimental::GlobalCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::GlobalCoordinateSystemTag >";


#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::LocalCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<double>,ROOT::Experimental::LocalCoordinateSystemTag >"    \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::LocalCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<float>,ROOT::Experimental::LocalCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::LocalCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Float16_t>,ROOT::Experimental::LocalCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::LocalCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::LocalCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<double>,ROOT::Experimental::LocalCoordinateSystemTag >"    \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::LocalCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<float>,ROOT::Experimental::LocalCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::LocalCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Float16_t>,ROOT::Experimental::LocalCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::LocalCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::GlobalCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<double>,ROOT::Experimental::GlobalCoordinateSystemTag >"    \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::GlobalCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<float>,ROOT::Experimental::GlobalCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::GlobalCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Float16_t>,ROOT::Experimental::GlobalCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::GlobalCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::GlobalCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<double>,ROOT::Experimental::GlobalCoordinateSystemTag >"    \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::GlobalCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<float>,ROOT::Experimental::GlobalCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::GlobalCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Float16_t>,ROOT::Experimental::GlobalCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::GlobalCoordinateSystemTag >";



#pragma link C++ class    ROOT::Experimental::PxPyPzE4D<Double32_t>+;
#pragma read sourceClass="ROOT::Experimental::PxPyPzE4D<double>"    \
             targetClass="ROOT::Experimental::PxPyPzE4D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::PxPyPzE4D<float>"     \
             targetClass="ROOT::Experimental::PxPyPzE4D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::PxPyPzE4D<Float16_t>" \
             targetClass="ROOT::Experimental::PxPyPzE4D<Double32_t>";

#pragma link C++ class    ROOT::Experimental::PtEtaPhiE4D<Double32_t>+;
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiE4D<double>"    \
             targetClass="ROOT::Experimental::PtEtaPhiE4D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiE4D<float>"     \
             targetClass="ROOT::Experimental::PtEtaPhiE4D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiE4D<Float16_t>" \
             targetClass="ROOT::Experimental::PtEtaPhiE4D<Double32_t>";

#pragma link C++ class    ROOT::Experimental::PxPyPzM4D<Double32_t>+;
#pragma read sourceClass="ROOT::Experimental::PxPyPzM4D<double>"    \
             targetClass="ROOT::Experimental::PxPyPzM4D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::PxPyPzM4D<float>"     \
             targetClass="ROOT::Experimental::PxPyPzM4D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::PxPyPzM4D<Float16_t>" \
             targetClass="ROOT::Experimental::PxPyPzM4D<Double32_t>";

#pragma link C++ class    ROOT::Experimental::PtEtaPhiM4D<Double32_t>+;
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiM4D<double>"    \
             targetClass="ROOT::Experimental::PtEtaPhiM4D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiM4D<float>"     \
             targetClass="ROOT::Experimental::PtEtaPhiM4D<Double32_t>";
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiM4D<Float16_t>" \
             targetClass="ROOT::Experimental::PtEtaPhiM4D<Double32_t>";


#pragma link C++ class    ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<Double32_t> >+;
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<double> >"    \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<float> >"     \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<Float16_t> >" \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<Double32_t> >";

#pragma link C++ class    ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<Double32_t> >+;
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<double> >"    \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<float> >"     \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<Float16_t> >" \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<Double32_t> >";

#pragma link C++ class    ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<Double32_t> >+;
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<double> >"    \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<float> >"     \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<Float16_t> >" \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<Double32_t> >";

#pragma link C++ class    ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<Double32_t> >+;
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<double> >"    \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<float> >"     \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<Double32_t> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<Float16_t> >" \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<Double32_t> >";




// #pragma link C++ typedef ROOT::Experimental::XYZVectorD32;
// #pragma link C++ typedef ROOT::Experimental::RhoEtaPhiVectorD32;
// #pragma link C++ typedef ROOT::Experimental::Polar3DVectorD32;

// #pragma link C++ typedef ROOT::Experimental::XYZPointD32;
// #pragma link C++ typedef ROOT::Experimental::RhoEtaPhiPointD32;
// #pragma link C++ typedef ROOT::Experimental::Polar3DPointD32;

// #pragma link C++ typedef ROOT::Experimental::XYZTVectorD32;
// #pragma link C++ typedef ROOT::Experimental::PtEtaPhiEVectorD32;
// #pragma link C++ typedef ROOT::Experimental::PxPyPzMVectorD32;
// #pragma link C++ typedef ROOT::Experimental::PtEtaPhiMVectorD32;



#endif
