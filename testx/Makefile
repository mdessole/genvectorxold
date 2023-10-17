EXTRALIBS        = -lGenVectorX 
LIBS = -L$(LIB) $(EXTRALIBS)
LIB = ../build/
CXXFLAGS += -g
INCLUDE = -I../inc

ObjSuf = o
SrcSuf = cpp
ExeSuf = 

MAINOBJ     = main.$(ObjSuf)
MAINSRC     = main.$(SrcSuf)
MAIN        = main$(ExeSuf)

#VECTORSCALEOBJ     = testVectorScale.$(ObjSuf)
#VECTORSCALESRC     = testVectorScale.$(SrcSuf)
#VECTORSCALE        = testVectorScale$(ExeSuf)


OBJS          = $(MAINOBJ) #$(COORDINATES4DOBJ) $(ROTATIONOBJ) $(BOOSTOBJ) $(GENVECTOROBJ) $(VECTORIOOBJ) $(STRESS3DOBJ) $(STRESS2DOBJ) $(ITERATOROBJ) $(VECTOROPOBJ) 


PROGRAMS      = $(MAIN)  #$(COORDINATES4D) $(ROTATION) $(BOOST) $(GENVECTOR) $(VECTORIO)  $(STRESS3D) $(STRESS2D) $(ITERATOR) $(VECTOROP) 

	  
.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)


all:            $(PROGRAMS)

$(MAIN):   $(MAINOBJ)
		    $(CXX) $(LDFLAGS) $(MAINOBJ) -o $@
		    @echo "$@ done"

clean:
		@rm -f $(OBJS)

distclean:      clean
		@rm -f $(PROGRAMS)


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $< $(LIBS) $(INCLUDE) 