Tracking validation
===================

The workhorse of the tracking validation is MultiTrackValidator, for
which documentation is available in
https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideMultiTrackValidator.

There is also a version of MultiTrackValidator for seeds called
TrackerSeedValidator, see
https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideTrackerSeedValidator
for more details.

The plotting tools are documented in
https://twiki.cern.ch/twiki/bin/view/CMS/TrackingValidationMC.


Ntuple
------

There is also an ntuple-version of MultiTrackValidator, called
[TrackingNtuple](plugins/TrackingNtuple.cc). It can be included in any
`cmsDriver.py`-generated workflow containing `VALIDATION` sequence by
including
`--customise Validation/RecoTrack/customiseTrackingNtuple.customiseTrackingNtuple`
argument to the `cmsDriver.py`. The customise function disables all
output modules and replaces the validation sequence with a sequence
producing the ntuple in `trackingNtuple.root` file. If ran without
RECO, it needs both RECO and DIGI files as an input.

For the ntuple content, take a look on the
[TrackingNtuple](plugins/TrackingNtuple.cc) code itself, and an
example PyROOT script for analysis,
[`trackingNtupleExample.py`](test/trackingNtupleExample.py).

By default the ntuple does not include hits or seeds. These can be
enabled with switches in
[`trackingNtuple_cff`](python/trackingNtuple_cff.py). Note that to
include seeds you have to run reconstruction as seeds are not stored
in RECO or AOD.