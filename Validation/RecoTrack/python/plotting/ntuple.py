import ROOT

class _Collection(object):
    def __init__(self, tree, sizeBranch, objclass):
        self._tree = tree
        self._sizeBranch = sizeBranch
        self._objclass = objclass

    def size(self):
        return int(getattr(self._tree, self._sizeBranch).size())

    def __len__(self):
        return self.size()

    def __getitem__(self, index):
        return self._objclass(self._tree, index)

class _Object(object):
    def __init__(self, tree, index, prefix):
        self._tree = tree
        self._index = index
        self._prefix = prefix

    def __getattr__(self, attr):
        self._checkIsValid()
        return lambda: getattr(self._tree, self._prefix+"_"+attr)[self._index]

    def _checkIsValid(self):
        if not self.isValid():
            raise Exception("%s is not valid" % self.__class__.__name__)

    def isValid(self):
        return self._index != -1

    def index(self):
        return self._index

    def nMatchedTrackingParticles(self):
        self._checkIsValid()
        return getattr(self._tree, self._prefix+"_simTrkIdx")[self._index].size()

    def matchedTrackingParticles(self):
        self._checkIsValid()
        for itp in getattr(self._tree, self._prefix+"_simTrkIdx")[self._index]:
            yield TrackingParticle(self._tree, itp)

class _HitObject(_Object):
    def __init__(self, tree, index, prefix):
        super(_HitObject, self).__init__(tree, index, prefix)

    def ntracks(self):
        self._checkIsValid()
        return getattr(self._tree, self._prefix+"_trkIdx")[self._index].size()

    def tracks(self):
        self._checkIsValid()
        for itrack in getattr(self._tree, self._prefix+"_trkIdx")[self._index]:
            yield Track(self._tree, itrack)

    def nseeds(self):
        self._checkIsValid()
        return getattr(self._tree, self._prefix+"_seeIdx")[self._index].size()

    def seeds(self):
        self._checkIsValid()
        for iseed in getattr(self._tree, self._prefix+"_seeIdx")[self._index]:
            yield Seed(self._tree, iseed)


class _HitAdaptor(object):
    def _hits(self):
        self._checkIsValid()
        for ihit, hitType in zip(self.hitIdx(), self.hitType()):
            yield (ihit, hitType)

    def hits(self):
        for ihit, hitType in self._hits():
            if hitType == 0:
                yield PixelHit(self._tree, ihit)
            elif hitType == 1:
                yield StripHit(self._tree, ihit)
            elif hitType == 2:
                yield GluedHit(self._tree, ihit)
            else:
                raise Exception("Unknown hit type %d" % hitType)

    def pixelHits(self):
        self._checkIsValid()
        for ihit, hitType in self._hits():
            if hitType != 0:
                continue
            yield PixelHit(self._tree, ihit)

    def stripHits(self):
        self._checkIsValid()
        for ihit, hitType in self._hits():
            if hitType != 1:
                continue
            yield StripHit(self._tree, ihit)

    def gluedHits(self):
        self._checkIsValid()
        for ihit, hitType in self._hits():
            if hitType != 2:
                continue
            yield GluedHit(self._tree, ihit)

##########
class TrackingNtuple(object):
    def __init__(self, fileName, tree="trackingNtuple/tree"):
        self._file = ROOT.TFile.Open(fileName)
        self._tree = self._file.Get(tree)
        self._entries = self._tree.GetEntriesFast()

    def tree(self):
        return self._tree

    def nevents(self):
        return self._entries

    def hasHits(self):
        return hasattr(self._tree, "pix_isBarrel")

    def hasSeeds(self):
        return hasattr(self._tree, "see_fitok")

    def __iter__(self):
        for jentry in xrange(self._entries):
            # get the next tree in the chain and verify
            ientry = self._tree.LoadTree( jentry )
            if ientry < 0: break
            # copy next entry into memory and verify
            nb = self._tree.GetEntry( jentry )
            if nb <= 0: continue

            yield Event(self._tree, jentry)

##########
class Event(object):
    def __init__(self, tree, entry):
        self._tree = tree
        self._entry = entry

    def entry(self):
        return self._entry

    def event(self):
        return self._tree.event

    def lumi(self):
        return self._tree.lumi

    def run(self):
        return self._tree.run

    def tracks(self):
        return Tracks(self._tree)

    def pixelHits(self):
        return PixelHits(self._tree)

    def stripHits(self):
        return StripHits(self._tree)

    def gluedHits(self):
        return GluedHits(self._tree)

    def seeds(self):
        return Seeds(self._tree)

    def trackingParticles(self):
        return TrackingParticles(self._tree)

    def vertices(self):
        return Vertices(self._tree)

    def trackingVertices(self):
        return TrackingVertices(self._tree)

##########
class BeamSpot(object):
    def __init__(self, tree):
        self._tree = tree
        self._prefix = "bsp"

    def __getattr__(self, attr):
        return lambda: getattr(self._tree, self._prefix+"_"+attr)

##########
class Track(_Object, _HitAdaptor):
    def __init__(self, tree, index):
        super(Track, self).__init__(tree, index, "trk")

    def nMatchedTrackingParticles(self):
        self._checkIsValid()
        return self._tree.trk_simIdx[self._index].size()

    def matchedTrackingParticles(self):
        self._checkIsValid()
        for isim in self._tree.trk_simIdx[self._index]:
            yield TrackingParticle(self._tree, isim)

    def seed(self):
        self._checkIsValid()
        return Seed(self._tree, self._tree.trk_seedIdx[self._index])

class Tracks(_Collection):
    def __init__(self, tree):
        super(Tracks, self).__init__(tree, "trk_pt", Track)

    def __iter__(self):
        for itrk in xrange(self.size()):
            yield Track(self._tree, itrk)

##########
class PixelHit(_HitObject):
    def __init__(self, tree, index):
        super(PixelHit, self).__init__(tree, index, "pix")

    def layerStr(self):
        if not self.isValid():
            return "Invalid"
        if self._tree.pix_isBarrel[self._index]:
            subdet = "BPix"
        else:
            subdet = "FPix"
        return "%s%d" % (subdet, self._tree.pix_lay[self._index])

class PixelHits(_Collection):
    def __init__(self, tree):
        super(PixelHits, self).__init__(tree, "pix_isBarrel", PixelHit)

    def __iter__(self):
        for ipix in xrange(self.size()):
            yield PixelHit(self._tree, ipix)

##########
class StripHit(_HitObject):
    def __init__(self, tree, index):
        super(StripHit, self).__init__(tree, index, "str")

    def layerStr(self):
        if not self.isValid():
            return "Invalid"
        return "%s%d" % ({3: "TIB", 4: "TID", 5: "TOB", 6: "TEC"}[self._tree.str_det[self._index]],
                         self._tree.str_lay[self._index])

class StripHits(_Collection):
    def __init__(self, tree):
        super(StripHits, self).__init__(tree, "str_isBarrel", StripHit)

    def __iter__(self):
        for istr in xrange(self.size()):
            yield StripHit(self._tree, istr)

##########
class GluedHit(_Object):
    def __init__(self, tree, index):
        super(GluedHit, self).__init__(tree, index, "glu")

    def monoHit(self):
        self._checkIsValid()
        return StripHit(self._tree, self._tree.glu_monoIdx[self._index])

    def stereoHit(self):
        self._checkIsValid()
        return StripHit(self._tree, self._tree.glu_stereoIdx[self._index])

    def nseeds(self):
        self._checkIsValid()
        return self._tree.glu_seeIdx[self._index].size()

    def seeds(self):
        self._checkIsValid()
        for iseed in self._tree.glu_seeIdx[self._index]:
            yield Seed(self._tree, iseed)

class GluedHits(_Collection):
    def __init__(self, tree):
        super(GluedHits, self).__init__(tree, "glu_isBarrel", GluedHit)

    def __iter__(self):
        for iglu in xrange(self.size()):
            yield GluedHit(self._tree, iglu)


##########
def _seedOffsetForAlgo(tree, algo):
    for ioffset, offset in enumerate(tree.see_offset):
        if tree.see_algo[offset] == algo:
            next_offset = tree.see_offset[ioffset+1] if ioffset < tree.see_offset.size()-1 else tree.see_algo.size()
            return (offset, next_offset)
    return (-1, -1)

class Seed(_Object, _HitAdaptor):
    def __init__(self, tree, index):
        super(Seed, self).__init__(tree, index, "see")

    def nMatchedTrackingParticles(self):
        self._checkIsValid()
        return self._tree.see_simIdx[self._index].size()

    def matchedTrackingParticles(self):
        self._checkIsValid()
        for isim in self._tree.see_simIdx[self._index]:
            yield TrackingParticle(self._tree, isim)

    def indexWithinAlgo(self):
        self._checkIsValid()
        algo = self._tree.see_algo[self._index]
        (offset, next_offset) = _seedOffsetForAlgo(self._tree, algo)
        if offset == -1: # algo not found
            return -1
        return self._index - offset

class Seeds(_Collection):
    def __init__(self, tree):
        super(Seeds, self).__init__(tree, "see_pt", Seed)

    def __iter__(self):
        for isee in xrange(self.size()):
            yield Seed(self._tree, isee)

    def nSeedsForAlgo(self, algo):
        (offset, next_offset) = _seedOffsetForAlgo(self._tree, algo)
        return next_offset - offset

    def seedsForAlgo(self, algo):
        (offset, next_offset) = _seedOffsetForAlgo(self._tree, algo)
        for isee in xrange(offset, next_offset):
            yield Seed(self._tree, isee)

##########
class TrackingParticle(_Object, _HitAdaptor):
    def __init__(self, tree, index):
        super(TrackingParticle, self).__init__(tree, index, "sim")

    def nMatchedTracks(self):
        self._checkIsValid()
        return self._tree.sim_trkIdx[self._index].size()

    def matchedTracks(self):
        self._checkIsValid()
        for itrk in self._tree.sim_trkIdx[self._index]:
            yield Track(self._tree, itrk)

    def parentVertex(self):
        self._checkIsValid()
        return TrackingVertex(self._tree, self._tree.sim_parentVtxIdx[self._index])

    def decayVertices(self):
        self._checkIsValid()
        for ivtx in self._tree.sim_decayVtxIdx[self._index]:
            yield TrackingVertex(self._tree, ivtx)

class TrackingParticles(_Collection):
    def __init__(self, tree):
        super(TrackingParticles, self).__init__(tree, "sim_pt", TrackingParticle)
        
    def __iter__(self):
        for isim in xrange(self.size()):
            yield TrackingParticle(self._tree, isim)

##########
class Vertex(_Object):
    def __init__(self, tree, index):
        super(Vertex, self).__init__(tree, index, "vtx")

    def nTracks(self):
        self._checkIsValid()
        return self._tree.vtx_trkIdx[self._index].size()

    def tracks(self):
        self._checkIsValid()
        for itrk in self._tree.vtx_trkIdx[self._index]:
            yield Track(self._tree, itrk)

class Vertices(_Collection):
    def __init__(self, tree):
        super(Vertices, self).__init__(tree, "vtx_valid", Vertex)

    def __iter__(self):
        for ivtx in xrange(self.size()):
            yield Vertex(self._tree, ivtx)


##########
class TrackingVertex(_Object):
    def __init__(self, tree, index):
        super(TrackingVertex, self).__init__(tree, index, "simvtx")

    def nSourceTrackingParticles(self):
        self._checkIsValid()
        return self._tree.simvtx_sourceSimIdx[self._index].size()

    def nDaughterTrackingParticles(self):
        self._checkIsValid()
        return self._tree.simvtx_daughterSimIdx[self._index].size()

    def sourceTrackingParticles(self):
        self._checkIsValid()
        for isim in self._tree.simvtx_sourceSimIdx[self._index]:
            yield TrackingParticle(self._tree, isim)

    def daughterTrackingParticles(self):
        self._checkIsValid()
        for isim in self._tree.simvtx_daughterSimIdx[self._index]:
            yield TrackingParticle(self._tree, isim)

class TrackingVertices(_Collection, TrackingVertex):
    def __init__(self, tree):
        super(TrackingVertex, self).__init__(tree, "simvtx_x")

    def __iter__(self):
        for isim in xrange(self.size()):
            yield TrackingVertex(self._tree, isim)
