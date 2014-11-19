import sys
import math

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Flag to indicate if it is ok to have missing files or histograms
# Set to true e.g. if there is no reference histograms
missingOk = False

class AlgoOpt:
    """Class to allow algorithm-specific values for e.g. plot bound values"""
    def __init__(self, default, **kwargs):
        """Constructor.

        Arguments:
        default -- default value

        Keyword arguments are treated as a dictionary where the key is a name of an algorithm, and the value is a value for that algorithm
        """
        self._default = default
        self._values = {}
        self._values.update(kwargs)

    def value(self, algo):
        """Get a value for an algorithm."""
        if algo in self._values:
            return self._values[algo]
        return self._default

def _getYmaxWithError(th1):
    return max([th1.GetBinContent(i)+th1.GetBinError(i) for i in xrange(1, th1.GetNbinsX()+1)])

def _findBounds(th1s, xmin=None, xmax=None, ymin=None, ymax=None):
    """Find x-y axis boundaries encompassing a list of TH1s if the bounds are not given in arguments.

    Arguments:
    th1s -- List of TH1s

    Keyword arguments:
    xmin -- Minimum x value; if None, take the minimum of TH1s
    xmax -- Maximum x value; if None, take the maximum of TH1s
    xmin -- Minimum y value; if None, take the minimum of TH1s
    xmax -- Maximum y value; if None, take the maximum of TH1s
    """
    if xmin is None or xmax is None or ymin is None or ymax is None or isinstance(ymax, list):
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        for th1 in th1s:
            xaxis = th1.GetXaxis()
            xmins.append(xaxis.GetBinLowEdge(xaxis.GetFirst()))
            xmaxs.append(xaxis.GetBinUpEdge(xaxis.GetLast()))
            ymins.append(th1.GetMinimum())
            ymaxs.append(th1.GetMaximum())
#            ymaxs.append(_getYmaxWithError(th1))

        if xmin is None:
            xmin = min(xmins)
        if xmax is None:
            xmax = max(xmaxs)
        if ymin is None:
            ymin = min(ymins)
        if ymax is None:
            ymax = 1.05*max(ymaxs)
        elif isinstance(ymax, list):
            ym = max(ymaxs)
            ymax = min(filter(lambda y: y>ym, ymax))

    for th1 in th1s:
        th1.GetXaxis().SetRangeUser(xmin, xmax)
        th1.GetYaxis().SetRangeUser(ymin, ymax)

    return (xmin, ymin, xmax, ymax)


class Efficiency:
    """Class to calculate efficiency from TH1s"""
    def __init__(self, name, numer, denom, numer2=None, title="", isEff=True):
        """Constructor.

        Arguments:
        name  -- String for the name of the resulting efficiency histogram
        numer -- String for the name of the numerator histogram
        denom -- String for the name of the denominator histogram

        Keyword arguments:
        numer2 -- String for the name of a second numerator histogram (default None).
        title  -- String for a title of the resulting histogram (default "")
        isEff  -- Boolean denoting if the class should calculate efficiency (default True)


        If numer2 is given, the numerator is (numer - numer2).

        If isEff is False, the "efficiency" is (1 - numerator) / denominator
        """
        self._name = name
        self._numer = numer
        self._numer2 = numer2
        self._denom = denom
        self._title = title
        self._isEff = isEff

    def __str__(self):
        """String representation, returns the name"""
        return self._name

    def create(self, tdirectory):
        """Create and return the efficiency histogram from a TDirectory"""
        # Get the numerator/denominator histograms
        hnum = tdirectory.Get(self._numer)
        hnum2 = tdirectory.Get(self._numer2) if self._numer2 is not None else None
        hdenom = tdirectory.Get(self._denom)

        # Check that they exist
        global missingOk
        if not hnum:
            print "Did not find {histo} from {dir}".format(histo=self._numer, dir=tdirectory.GetPath())
            if missingOk:
                return None
            else:
                sys.exit(1)
        if self._numer2 and not hnum2:
            print "Did not find {histo} from {dir}".format(histo=self._numer2, dir=tdirectory.GetPath())
            if missingOk:
                return None
            else:
                sys.exit(1)
        if not hdenom:
            print "Did not find {histo} from {dir}" .format(histo=self._denom, dir=tdirectory.GetPath())
            if missingOk:
                return None
            else:
                sys.exit(1)
                    
        heff = hdenom.Clone(self._name)
        heff.SetTitle(self._title)

        # Function to calculate the efficiency
        if self._isEff:
            _effVal = lambda numerVal, denomVal: numerVal / denomVal if denomVal != 0.0 else 0.0
        else:
            _effVal = lambda numerVal, denomVal: (1 - numerVal / denomVal) if denomVal != 0.0 else 0.0

        # Function to get the numerator
        if self._numer2 is None:
            _numerVal = lambda i, num, num2: num.GetBinContent(i)
        else:
            _numerVal = lambda i, num, num2: num.GetBinContent(i) - num2.GetBinContent(i)

        # Calculate efficiencies for each bin
        for i in xrange(1, hdenom.GetNbinsX()+1):
            numerVal = _numerVal(i, hnum, hnum2)
            denomVal = hdenom.GetBinContent(i)

            effVal = _effVal(numerVal, denomVal)
            errVal = math.sqrt(effVal*(1-effVal)/denomVal) if (denomVal != 0.0 and effVal <= 1) else 0.0

            heff.SetBinContent(i, effVal)
            heff.SetBinError(i, errVal)

        return heff

class AggregateBins:
    def __init__(self, name, histoName, mapping, normalizeTo=None):
        self._name = name
        self._histoName = histoName
        self._mapping = mapping
        self._normalizeTo = normalizeTo

    def __str__(self):
        return self._name

    def create(self, tdirectory):
        th1 = tdirectory.Get(self._histoName)

        global missingOk
        if not th1:
            print "Did not find {histo} from {dir}".format(histo=self._histoName, dir=tdirectory.GetPath())
            if missingOk:
                return None
            else:
                sys.exit(1)

        result = ROOT.TH1F(self._name, self._name, len(self._mapping), 0, len(self._mapping))

        for i, (key, labels) in enumerate(self._mapping.iteritems()):
            sumTime = 0
            for l in labels:
                bin = th1.GetXaxis().FindBin(l)
                if bin > 0:
                    sumTime += th1.GetBinContent(bin)
            result.SetBinContent(i+1, sumTime)
            result.GetXaxis().SetBinLabel(i+1, key)

        if self._normalizeTo is not None:
            bin = th1.GetXaxis().FindBin(self._normalizeTo)
            if bin <= 0:
                print "Trying to normalize {name} to {binlabel}, which does not exist".format(name=self._name, binlabel=self._normalizeTo)
                sys.exit(1)
            value = th1.GetBinContent(bin)
            if value != 0:
                result.Scale(1/value)

        for bin in xrange(1, result.GetNbinsX()+1):
            print "%s %f" % (result.GetXaxis().GetBinLabel(bin), result.GetBinContent(bin))

        return result

# Plot styles
_plotStylesColor = [4, 2, ROOT.kOrange+7, ROOT.kMagenta-3]
_plotStylesMarker = [21, 20, 22, 34]

class Plot:
    """Represents one plot, comparing one or more histograms."""
    def __init__(self, name, **kwargs):
        """ Constructor.

        Arguments:
        name -- String for name of the plot, or Efficiency object

        Keyword arguments:
        title        -- String for a title of the plot (default None)
        xtitle       -- String for x axis title (default None)
        ytitle       -- String for y axis title (default None)
        ytitlesize   -- Float for y axis title size (default None)
        ytitleoffset -- Float for y axis title offset (default None)
        xmin         -- Float for x axis minimum (default None, i.e. automatic)
        xmax         -- Float for x axis maximum (default None, i.e. automatic)
        ymin         -- Float for y axis minimum (default 0)
        ymax         -- Float for y axis maximum (default None, i.e. automatic)
        xlog         -- Bool for x axis log status (default False)
        ylog         -- Bool for y axis log status (default False)
        xgrid        -- Bool for x axis grid status (default True)
        ygrid        -- Bool for y axis grid status (default True)
        stat         -- Draw stat box? (default False)
        fit          -- Do gaussian fit? (default False)
        normalizeToUnitArea -- Normalize histograms to unit area? (default False)
        profileX     -- Take histograms via ProfileX()? (default False)
        fitSlicesY   -- Take histograms via FitSlicesY() (default False)
        scale        -- Scale histograms by a number (default None)
        xbinlabels   -- List of x axis bin labels (if given, default None)
        xbinlabelsize -- Size of x axis bin labels (default None)
        xbinlabeloption -- Option string for x axis bin labels (default None)
        drawStyle    -- If "hist", draw as line instead of points (default None)
        lineWidth    -- If drawStyle=="hist", the width of line (default 2)
        legendDx     -- Float for moving TLegend in x direction for separate=True (default None)
        legendDy     -- Float for moving TLegend in y direction for separate=True (default None)
        legendDw     -- Float for changing TLegend width for separate=True (default None)
        legendDh     -- Float for changing TLegend height for separate=True (default None)
        """
        self._name = name

        def _set(attr, default):
            setattr(self, "_"+attr, kwargs.get(attr, default))

        _set("title", None)
        _set("xtitle", None)
        _set("ytitle", None)
        _set("ytitlesize", None)
        _set("ytitleoffset", None)

        _set("xmin", None)
        _set("xmax", None)
        _set("ymin", 0.)
        _set("ymax", None)

        _set("xlog", False)
        _set("ylog", False)
        _set("xgrid", True)
        _set("ygrid", True)

        _set("stat", False)
        _set("fit", False)

        _set("statx", 0.65)
        _set("staty", 0.8)

        _set("normalizeToUnitArea", False)
        _set("profileX", False)
        _set("fitSlicesY", False)
        _set("scale", None)
        _set("xbinlabels", None)
        _set("xbinlabelsize", None)
        _set("xbinlabeloption", None)

        _set("drawStyle", None)
        _set("lineWidth", 2)

        _set("legendDx", None)
        _set("legendDy", None)
        _set("legendDw", None)
        _set("legendDh", None)

        self._histograms = []

    def getNumberOfHistograms(self):
        """Return number of existing histograms."""
        return len(self._histograms)

    def getName(self):
        return str(self._name)

    def _createOne(self, tdir):
        """Create one histogram from a TDirectory."""
        if tdir == None:
            return None

        # If name is Efficiency instead of string, call its create()
        if hasattr(self._name, "create"):
            th1 = self._name.create(tdir)
        else:
            th1 = tdir.Get(self._name)

        # Check the histogram exists
        if th1 == None:
            print "Did not find {histo} from {dir}".format(histo=self._name, dir=tdir.GetPath())
            if missingOk:
                return None
            else:
                sys.exit(1)

        if self._profileX:
            th1 = th1.ProfileX()

        if self._fitSlicesY:
            ROOT.TH1.AddDirectory(True)
            th1.FitSlicesY()
            th1 = ROOT.gDirectory.Get(th1.GetName()+"_2")
            th1.SetDirectory(None)
            #th1.SetName(th1.GetName()+"_ref")
            ROOT.TH1.AddDirectory(False)

        if self._title is not None:
            th1.SetTitle(self._title)

        if self._scale is not None:
            th1.Scale(self._scale)

        return th1

    def create(self, tdirs):
        """Create histograms from list of TDirectories"""
        if len(tdirs) > len(_plotStylesColor):
            raise Exception("More TDirectories than there are plot styles defined. Please define more plot styles in this file")

        self._histograms = [self._createOne(tdir) for tdir in tdirs]

    def _setStats(self, startingX, startingY):
        """Set stats box."""
        if not self._stat:
            for h in self._histograms:
                if h is not None:
                    h.SetStats(0)
            return

        def _doStats(h, col, dy):
            if h is None:
                return
            h.SetStats(1)

            if self._fit:
                h.Fit("gaus", "Q")
                f = h.GetListOfFunctions().FindObject("gaus")
                if f == None:
                    h.SetStats(0)
                    return
                f.SetLineColor(col)
                f.SetLineWidth(1)
            h.Draw()
            ROOT.gPad.Update()
            st = h.GetListOfFunctions().FindObject("stats")
            if self._fit:
                st.SetOptFit(0010)
                st.SetOptStat(1001)
            st.SetX1NDC(startingX)
            st.SetX2NDC(startingX+0.3)
            st.SetY1NDC(startingY+dy)
            st.SetY2NDC(startingY+dy+0.15)
            st.SetTextColor(col)

        dy = 0.0
        for i, h in enumerate(self._histograms):
            _doStats(h, _plotStylesColor[i], dy)
            dy -= 0.19

    def _normalize(self):
        """If >= 2 histograms, normalize them to unit area."""
        if len(self._histograms) < 2:
            return

        integrals = []
        n_nonzero = 0
        for h in self._histograms:
            if h is None:
                continue
            i = h.Integral()
            if i != 0:
                n_nonzero += 1
            integrals.append(i)

        if n_nonzero >= 2:
            for i, h in zip(integrals, self._histograms):
                if h is None:
                    continue
                if i != 0:
                    h.Scale(1.0/i)

    def draw(self, algo):
        """Draw the histograms using values for a given algorithm."""
        if self._normalizeToUnitArea:
            self._normalize()

        def _styleMarker(h, msty, col):
            h.SetMarkerStyle(msty)
            h.SetMarkerColor(col)
            h.SetMarkerSize(0.7)
            h.SetLineColor(1)
            h.SetLineWidth(1)

        def _styleHist(h, msty, col):
            _styleMarker(h, msty, col)
            h.SetLineColor(col)
            h.SetLineWidth(self._lineWidth)

        # Use marker or hist style
        style = _styleMarker
        if self._drawStyle is not None:
            if "hist" in self._drawStyle.lower():
                style = _styleHist

        # Apply style to histograms, filter out Nones
        histos = []
        for i, h in enumerate(self._histograms):
            if h is None:
                continue
            style(h, _plotStylesMarker[i], _plotStylesColor[i])
            histos.append(h)
        if len(histos) == 0:
            print "No histograms for plot {name}".format(name=self._name)
            return

        # Set log and grid
        ROOT.gPad.SetLogx(self._xlog)
        ROOT.gPad.SetLogy(self._ylog)
        ROOT.gPad.SetGridx(self._xgrid)
        ROOT.gPad.SetGridy(self._ygrid)

        # Return value if number, or algo-specific value if AlgoOpt
        def _getVal(val):
            if hasattr(val, "value"):
                return val.value(algo)
            return val

        bounds = _findBounds(histos,
                             xmin=_getVal(self._xmin), xmax=_getVal(self._xmax),
                             ymin=_getVal(self._ymin), ymax=_getVal(self._ymax))

        # Create bounds before stats in order to have the
        # SetRangeUser() calls made before the fit
        self._setStats(self._statx, self._staty)

        xbinlabels = self._xbinlabels
        if xbinlabels is None:
            if len(histos[0].GetXaxis().GetBinLabel(1)) > 0:
                xbinlabels = []
                for i in xrange(1, histos[0].GetNbinsX()+1):
                    xbinlabels.append(histos[0].GetXaxis().GetBinLabel(i))

        # Create frame
        if xbinlabels is None:
            frame = ROOT.gPad.DrawFrame(*bounds)
        else:
            # Special form needed if want to set x axis bin labels
            nbins = len(xbinlabels)
            frame = ROOT.TH1F("hframe", "hframe", nbins, bounds[0], bounds[2])
            frame.SetBit(ROOT.TH1.kNoStats)
            frame.SetBit(ROOT.kCanDelete)
            frame.SetMinimum(bounds[1])
            frame.SetMaximum(bounds[3])
            frame.GetYaxis().SetLimits(bounds[1], bounds[3])
            frame.Draw("")

            xaxis = frame.GetXaxis()
            for i in xrange(nbins):
                xaxis.SetBinLabel(i+1, xbinlabels[i])
            if self._xbinlabelsize is not None:
                xaxis.SetLabelSize(self._xbinlabelsize)
            if self._xbinlabeloption is not None:
                frame.LabelsOption(self._xbinlabeloption)

        # Set properties of frame
        frame.SetTitle(histos[0].GetTitle())
        if self._xtitle is not None:
            frame.GetXaxis().SetTitle(self._xtitle)
        if self._ytitle is not None:
            frame.GetYaxis().SetTitle(self._ytitle)
        if self._ytitlesize is not None:
            frame.GetYaxis().SetTitleSize(self._ytitlesize)
        if self._ytitleoffset is not None:
            frame.GetYaxis().SetTitleOffset(self._ytitleoffset)

        # Draw histograms
        for h in histos:
            h.Draw("sames")

        ROOT.gPad.RedrawAxis()
        self._frame = frame # keep the frame in memory for sure

    def addToLegend(self, legend, legendLabels):
        """Add histograms to a legend.

        Arguments:
        legend       -- TLegend
        legendLabels -- List of strings for the legend labels
        """
        for h, label in zip(self._histograms, legendLabels):
            if h is None:
                continue
            legend.AddEntry(h, label, "LPF")

class PlotGroup:
    """Group of plots, results a TCanvas"""
    def __init__(self, name, plots, **kwargs):
        """Constructor.

        Arguments:
        name  -- String for name of the TCanvas, used also as the basename of the picture files
        plots -- List of Plot objects

        Keyword arguments:
        legendDx -- Float for moving TLegend in x direction (default None)
        legendDy -- Float for moving TLegend in y direction (default None)
        legendDw -- Float for changing TLegend width (default None)
        legendDh -- Float for changing TLegend height (default None)
        """
        self._name = name
        self._plots = plots

        if len(self._plots) <= 2:
            self._canvas = ROOT.TCanvas(self._name, self._name, 1000, 500)
        elif len(self._plots) <= 4:
            self._canvas = ROOT.TCanvas(self._name, self._name, 1000, 1050)
        else:
            self._canvas = ROOT.TCanvas(self._name, self._name, 1000, 1400)

        self._canvasSingle = ROOT.TCanvas(self._name+"Single", self._name, 500, 500)
        # from TDRStyle
        self._canvasSingle.SetTopMargin(0.05)
        self._canvasSingle.SetBottomMargin(0.13)
        self._canvasSingle.SetLeftMargin(0.16)
        self._canvasSingle.SetRightMargin(0.05)


        def _set(attr, default):
            setattr(self, "_"+attr, kwargs.get(attr, default))

        _set("legendDx", None)
        _set("legendDy", None)
        _set("legendDw", None)
        _set("legendDh", None)

    def create(self, tdirectories):
        """Create histograms from a list of TDirectories."""
        for plot in self._plots:
            plot.create(tdirectories)

    def draw(self, algo, legendLabels, prefix=None, separate=False, saveFormat=".pdf"):
        """Draw the histograms using values for a given algorithm.

        Arguments:
        algo          -- string for algorithm
        legendLabels  -- List of strings for legend labels (corresponding to the tdirectories in create())
        prefix        -- Optional string for file name prefix (default None)
        separate      -- Save the plots of a group to separate files instead of a file per group (default False)
        saveFormat   -- String specifying the plot format (default '.pdf')
        """
        if separate:
            return self._drawSeparate(algo, legendLabels, prefix, saveFormat)

        self._canvas.Divide(2, int((len(self._plots)+1)/2)) # this should work also for odd n

        # Draw plots to canvas
        for i, plot in enumerate(self._plots):
            self._canvas.cd(i+1)
            plot.draw(algo)

        # Setup legend
        self._canvas.cd()
        if len(self._plots) <= 4:
            lx1 = 0.2
            lx2 = 0.9
            ly1 = 0.48
            ly2 = 0.53
        else:
            lx1 = 0.1
            lx2 = 0.9
            ly1 = 0.64
            ly2 = 0.69
        if self._legendDx is not None:
            lx1 += self._legendDx
            lx2 += self._legendDx
        if self._legendDy is not None:
            ly1 += self._legendDy
            ly2 += self._legendDy
        if self._legendDw is not None:
            lx2 += self._legendDw
        if self._legendDh is not None:
            ly1 -= self._legendDh
        plot = max(self._plots, key=lambda p: p.getNumberOfHistograms())
        legend = self._createLegend(plot, legendLabels, lx1, ly1, lx2, ly2)

        return self._save(self._canvas, saveFormat, prefix=prefix)

    def _drawSeparate(self, algo, legendLabels, prefix, saveFormat):
        lx1def = 0.6
        lx2def = 0.95
        ly1def = 0.85
        ly2def = 0.95

        ret = []

        for plot in self._plots:
            # Draw plot to canvas
            self._canvasSingle.cd()
            plot.draw(algo)

            # Setup legend
            lx1 = lx1def
            lx2 = lx2def
            ly1 = ly1def
            ly2 = ly2def

            if plot._legendDx is not None:
                lx1 += plot._legendDx
                lx2 += plot._legendDx
            if plot._legendDy is not None:
                ly1 += plot._legendDy
                ly2 += plot._legendDy
            if plot._legendDw is not None:
                lx2 += plot._legendDw
            if plot._legendDh is not None:
                ly1 -= plot._legendDh

            legend = self._createLegend(plot, legendLabels, lx1, ly1, lx2, ly2, textSize=0.03)

            ret.extend(self._save(self._canvasSingle, saveFormat, prefix=prefix, postfix="_"+plot.getName()))
        return ret

    def _createLegend(self, plot, legendLabels, lx1, ly1, lx2, ly2, textSize=0.016):
        l = ROOT.TLegend(lx1, ly1, lx2, ly2)
        l.SetTextSize(textSize)
        l.SetLineColor(1)
        l.SetLineWidth(1)
        l.SetLineStyle(1)
        l.SetFillColor(0)
        l.SetMargin(0.07)

        plot.addToLegend(l, legendLabels)
        l.Draw()
        return l

    def _save(self, canvas, saveFormat, prefix=None, postfix=None):
        # Save the canvas to file and clear
        name = self._name
        if prefix is not None:
            name = prefix+name
        if postfix is not None:
            name = name+postfix
        canvas.SaveAs(name+saveFormat)
        canvas.Clear()

        return [name+saveFormat]


class Plotter:
    """Represent a collection of PlotGroups."""
    def __init__(self, possibleDirs, plotGroups, saveFormat=".pdf"):
        """Constructor.

        Arguments:
        possibleDirs -- List of strings for possible directories of histograms in TFiles
        plotGroups   -- List of PlotGroup objects
        saveFormat   -- String specifying the plot format (default '.pdf')
        """
        self._possibleDirs = possibleDirs
        self._plotGroups = plotGroups
        self._saveFormat = saveFormat

        ROOT.gROOT.SetStyle("Plain")
        ROOT.gStyle.SetPadRightMargin(0.07)
        ROOT.gStyle.SetPadLeftMargin(0.13)
        ROOT.gStyle.SetTitleFont(42, "XYZ")
        ROOT.gStyle.SetTitleSize(0.05, "XYZ")
        ROOT.gStyle.SetTitleOffset(1.2, "Y")
        #ROOT.gStyle.SetTitleFontSize(0.05)
        ROOT.gStyle.SetLabelFont(42, "XYZ")
        ROOT.gStyle.SetLabelSize(0.05, "XYZ")
        ROOT.gStyle.SetTextSize(0.05)

        ROOT.TH1.AddDirectory(False)

    def setPossibleDirectoryNames(self, possibleDirs):
        self._possibleDirs = possibleDirs

    def getPossibleDirectoryNames(self):
        """Return the list of possible directory names."""
        return self._possibleDirs

    def _getDir(self, tfile, subdir):
        """Get TDirectory from TFile."""
        if tfile is None:
            return None
        for pd in self._possibleDirs:
            d = tfile.Get(pd)
            if d:
                if subdir is not None:
                    # Pick associator if given
                    d = d.Get(subdir)
                    if d:
                        return d
                    else:
                        msg = "Did not find subdirectory '%s' from directory '%s' in file %s" % (subdir, pd, tfile.GetName())
                        if missingOk:
                            print msg
                            return None
                        else:
                            raise Exception(msg)
                else:
                    return d
        msg = "Did not find any of directories '%s' from file %s" % (",".join(self._possibleDirs), tfile.GetName())
        if missingOk:
            print msg
            return None
        else:
            raise Exception(msg)

    def create(self, files, labels, subdir=None):
        """Create histograms from a list of TFiles.

        Arguments:
        files  -- List of TFiles
        labels -- List of strings for legend labels corresponding the files
        subdir -- Optional string for subdirectory inside the possibleDirs
        """
        dirs = []
        self._labels = []
        for f, l in zip(files, labels):
            d = self._getDir(f, subdir)
            dirs.append(d)
            self._labels.append(l)

        for pg in self._plotGroups:
            pg.create(dirs)

    def draw(self, algo, prefix=None, separate=False):
        """Draw and save all plots using settings of a given algorithm.

        Arguments:
        algo     -- String for the algorithm
        prefix   -- Optional string for file name prefix (default None)
        separate -- Save the plots of a group to separate files instead of a file per group (default False)
        """
        ret = []
        for pg in self._plotGroups:
            ret.extend(pg.draw(algo, self._labels, prefix=prefix, separate=separate, saveFormat=self._saveFormat))
        return ret

