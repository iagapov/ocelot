#! /usr/bin/env python
import sys, time

import sdds

from pylab import *


def plotOptics(idx=0,fname='twiss', limits=None):
     x = sdds.SDDS(0)
     x.load(fname + '.twi')
     #print(x.columnName)
     #print(x.columnName[0], x.columnName[1], x.columnName[6], x.columnName[7])
     #print(x.columnData[0])
     ax = plt.figure().add_subplot(111)
     s = x.columnData[0][idx]
     beta_x = np.array(x.columnData[1][idx])
     beta_y = np.array(x.columnData[7][idx])
     eta_x = np.array(x.columnData[4][idx])
     if limits == None:
         p1,=ax.plot(s, beta_x, 'b-', lw=3)
         p2,=ax.plot(s, beta_y, 'r-', lw=3)
         ax2 = ax.twinx()
         p3,=ax2.plot(s, eta_x, 'g-', lw=3)
     else:
         i1 = 0
         i2 = len(s)-1
         print(i2)
         for i in range(1,len(s)):
             if s[i-1] < limits[0] and s[i] >= limits[0]: i1 = i
             if s[i-1] < limits[1] and s[i] >= limits[1]: i2 = i

         p1,=ax.plot(s[i1:i2], beta_x[i1:i2], 'b-', lw=3)
         p2,=ax.plot(s[i1:i2], beta_y[i1:i2], 'r-', lw=3)
         ax2 = ax.twinx()
         p3,=ax2.plot(s[i1:i2], eta_x[i1:i2], 'g-', lw=3)
     ax.legend([p1,p2,p3],['$\\beta_x$','$\\beta_y$','$\\eta_x$'])
     plt.show()


def plotCoupledOptics(idx=0,fname='twiss', limits=None):
     x = sdds.SDDS(0)
     x.load(fname + '.ctwi')
     #print(x.columnName)
     #print(x.columnName[0], x.columnName[1], x.columnName[6], x.columnName[7])
     #print(x.columnData[0])
     ax = plt.figure().add_subplot(111)
     s = np.array(x.columnData[1][idx])
     beta_x1 = np.array(x.columnData[8][idx])
     beta_x2 = np.array(x.columnData[9][idx])
     beta_y1 = np.array(x.columnData[10][idx])
     beta_y2 = np.array(x.columnData[11][idx])
     eta_x = np.array(x.columnData[12][idx])

     if limits == None:
         p1,=ax.plot(s, beta_x1, 'b-', lw=3)
         p2,=ax.plot(s, beta_y1, 'r--', lw=3)
         ax.plot(s, beta_x2, 'b--', lw=2)
         ax.plot(s, beta_y2, 'r-', lw=2)
         ax2 = ax.twinx()
         p3,=ax2.plot(s, eta_x, 'g-', lw=3)
     else:
         i1 = 0
         i2 = len(s)-1
         print(i2)
         for i in range(1,len(s)):
             if s[i-1] < limits[0] and s[i] >= limits[0]: i1 = i
             if s[i-1] < limits[1] and s[i] >= limits[1]: i2 = i

         p1,=ax.plot(s[i1:i2], beta_x1[i1:i2], 'b-', lw=3)
         p2,=ax.plot(s[i1:i2], beta_y1[i1:i2], 'r-', lw=3)
         ax.plot(s[i1:i2], beta_x1[i1:i2], 'b--', lw=2)
         ax.plot(s[i1:i2], beta_y1[i1:i2], 'r--', lw=2)
         ax2 = ax.twinx()
         p3,=ax2.plot(s[i1:i2], eta_x[i1:i2], 'g-', lw=3)
     ax.legend([p1,p2,p3],['$\\beta_x$','$\\beta_y$','$\\eta_x$'])
     plt.show()


def get_s_dict(twiss_file):
    #print(x.columnName[0],x.columnName[14],x.columnName[15],x.columnName[16])

    n_list = len(twiss_file.columnData[0][0])

    s_dict = {}

    for idx in range(n_list):
        s = twiss_file.columnData[0][0][idx]
        ename = twiss_file.columnData[14][0][idx]
        eocc = twiss_file.columnData[15][0][idx]
        etype = twiss_file.columnData[16][0][idx]

        s_dict[ename, eocc] = s
    return s_dict


def showCorrectors(ipage=0,fname_calc='twiss', fname='twiss'):
    twiss_file = sdds.SDDS(0)
    twiss_file.load(fname_calc + '.twi')

    s_dict = get_s_dict(twiss_file)

    x2 = sdds.SDDS(0)
    x2.load(fname+'.param')

    #['ElementName', 'ElementParameter', 'ParameterValue', 'ElementType', 'ElementOccurence', 'ElementGroup']
    #print(x2.columnName[0], x2.columnName[1], x2.columnName[2], x2.columnName[3], x2.columnName[4])
    n_list = len(x2.columnData[0][ipage])

    s_a = []
    hc_a = []
    vc_a = []

    for idx in range(n_list):
        ename = x2.columnData[0][ipage][idx] # ElementName
        par = x2.columnData[1][ipage][idx] #ElementParameter
        eval = x2.columnData[2][ipage][idx] #ParameterValue
        etype = x2.columnData[3][ipage][idx] #ElementType
        eocc = x2.columnData[4][ipage][idx] #ElementOcuurance
        if etype in ['KICKER']:
            if par == "HKICK":
                hc_a.append(eval)
                s_a.append(s_dict[ename, eocc])
            if par == "VKICK":
                vc_a.append(eval)


    ax = plt.figure().add_subplot(111)
    ax.plot(s_a, 1.e3*np.array(hc_a), 'bd-')
    ax.plot(s_a, 1.e3*np.array(vc_a), 'gd--')
    ax.set_xlabel('$s \, (m)$')
    ax.set_ylabel('$Corrector kick$ ($m rad$)')
    plt.show()
    return 1.e3*np.array(hc_a), 1.e3*np.array(vc_a)

def showQuads(ipage=0,fname_calc='twiss', fname='twiss'):
    twiss_file = sdds.SDDS(0)
    twiss_file.load(fname_calc + '.twi')

    s_dict = get_s_dict(twiss_file)

    x2 = sdds.SDDS(0)
    x2.load(fname+'.param')

    #['ElementName', 'ElementParameter', 'ParameterValue', 'ElementType', 'ElementOccurence', 'ElementGroup']
    #print(x2.columnName[0], x2.columnName[1], x2.columnName[2], x2.columnName[3], x2.columnName[4])
    n_list = len(x2.columnData[0][ipage])

    s_a = []
    k1_a = []

    for idx in range(n_list):
        ename = x2.columnData[0][ipage][idx] # ElementName
        par = x2.columnData[1][ipage][idx] #ElementParameter
        eval = x2.columnData[2][ipage][idx] #ParameterValue
        etype = x2.columnData[3][ipage][idx] #ElementType
        eocc = x2.columnData[4][ipage][idx] #ElementOcuurance
        if etype in ['KQUAD']:
            if par == "K1":
                k1_a.append(eval)
                s_a.append(s_dict[ename, eocc])


    ax = plt.figure().add_subplot(111)
    ax.plot(s_a, np.array(k1_a), 'rd')
    ax.set_xlabel('$s \, (m)$')
    ax.set_ylabel('k1')
    plt.show()

def get_quad_families(fname = 'twiss'):
    #twiss_file = sdds.SDDS(0)
    #twiss_file.load(fname + '.twi')
    x2 = sdds.SDDS(0)
    x2.load(fname+'.param')
    ipage = 0
    #['ElementName', 'ElementParameter', 'ParameterValue', 'ElementType', 'ElementOccurence', 'ElementGroup']
    #print(x2.columnName[0], x2.columnName[1], x2.columnName[2], x2.columnName[3], x2.columnName[4])
    n_list = len(x2.columnData[0][ipage])

    eocc_a = {}
    vals = {}

    for idx in range(n_list):
        ename = x2.columnData[0][ipage][idx] # ElementName
        par = x2.columnData[1][ipage][idx] #ElementParameter
        eval = x2.columnData[2][ipage][idx] #ParameterValue
        etype = x2.columnData[3][ipage][idx] #ElementType
        eocc = x2.columnData[4][ipage][idx] #ElementOcuurance
        if etype in ['KQUAD']:
            if par == "K1":
                eocc_a[ename] = int(eocc)
                vals[ename, eocc] = float(eval)

    return eocc_a, vals


def plotOrb(idx=0, fname='twiss'):
     x = sdds.SDDS(0)
     x.load(fname + '.clo')

     ax = plt.figure().add_subplot(111)
     p1, = ax.plot(x.columnData[0][idx], 1.e6*np.array(x.columnData[1][idx]), 'b-',lw=3)
     p2, = ax.plot(x.columnData[0][idx], 1.e6*np.array(x.columnData[3][idx]), 'r-',lw=3)
     ax.set_xlabel('$s \, (m)$')
     ax.set_ylabel('Orbit ($\mu m$)')
     ax.legend([p1,p2],['X','Y'])

     print('rms x [mu m]: ', 1.e6*np.std(np.array(x.columnData[1][idx])))
     print('rms y [mu m]: ', 1.e6*np.std(np.array(x.columnData[3][idx])))
     print('max x [mu m]: ', 1.e6*np.max(np.array(x.columnData[1][idx])))
     print('max y [mu m]: ', 1.e6*np.max(np.array(x.columnData[3][idx])))


     plt.show()


# orbit respoonse matrix
class ORM:
    def __init__(self, n_cor, n_bpm):
        hcors = []
        vcors = []
        self.Rxx = np.zeros([n_cor,n_bpm])
        self.Ryy = np.zeros([n_cor,n_bpm])
        self.Rxy = np.zeros([n_cor,n_bpm])
        self.Ryx = np.zeros([n_cor,n_bpm])

# orbit response data
class OrbitResponse:
    def __init__(self):
        self.hcors = []
        self.orbits = {}
        self.scan_values = {}

def get_bpms(sdds_file, ipage):
    n_list = len(sdds_file.columnData[0][ipage]) # length
    #print(sdds_file.columnName)
    #print('orbit data length length:', n_list)
    xs = []
    ys = []
    for idx in range(n_list):
        s = sdds_file.columnData[0][ipage][idx]
        x = sdds_file.columnData[1][ipage][idx]
        y = sdds_file.columnData[3][ipage][idx]
        etype = sdds_file.columnData[7][ipage][idx]
        #print(etype)
        if etype in ['MONI']:
            #print('BPM!!!')
            xs.append(x)
            ys.append(y)

    return np.array(xs), np.array(ys)

def get_quads():
    pass

def get_bpm_response(orbits, ref_orbit, i_bpm, dkick):

    x = [ orbits[0][i_bpm] , ref_orbit[0][i_bpm]  ]# i is ipage
    y = [ orbits[1][i_bpm] ,ref_orbit[1][i_bpm] ]

    cx = (x[0] - x[1]) / dkick # coefficient, first order
    cy = (y[0] - y[1]) / dkick # coefficient, first order
    return cx,cy

# stub, return default scan range
def get_scan_values(cor, param_file, n_pages):
    #cor = fname.replace('orm_x','')

    print('getting scan values for', cor)

    ipage = 0
    n_list = len(param_file.columnData[0][ipage])
    vx_a = []
    vy_a = []

    for idx in range(n_list):
        for ipage in range(n_pages):
            ename = param_file.columnData[0][ipage][idx] # ElementName
            par = param_file.columnData[1][ipage][idx] #ElementParameter
            eval = param_file.columnData[2][ipage][idx] #ParameterValue
            etype = param_file.columnData[3][ipage][idx] #ElementType
            eocc = param_file.columnData[4][ipage][idx] #ElementOcuurance
            if etype in ['KICKER'] and ename.lower() == cor.lower():
                if par == "HKICK":
                    vx_a.append(eval)
                if par == "VKICK":
                    vy_a.append(eval)

    return np.array(vx_a), np.array(vy_a)

# build ORM from elegant sdds files, simulation of measurement
# units are [m/rad] or [mm/mrad]
def build_orm_old(fnames, n_pages, plane='x'):
    orb_file = sdds.SDDS(0)
    param_file  = sdds.SDDS(0)

    ore = OrbitResponse()
    orb_file.load(fnames[0] + '.clo')
    x_, y_ =  get_bpms(orb_file, ipage=0)
    n_bpms = len(x_)

    print('building ORM', len(fnames), n_bpms)

    orm = ORM(len(fnames), n_bpms)

    i_cor = 0
    for fname in fnames:
        orb_file.load(fname + '.clo')
        param_file.load(fname + '.param')
        print(orb_file.columnName)
        ore.hcors.append(fname)
        ore.orbits[fname] = {}
        cname = fname.replace('orm_'+plane+'_','')
        vx, vy = get_scan_values(cname, param_file, n_pages)
        if plane == 'x' : ore.scan_values[fname] = vx
        if plane == 'y' : ore.scan_values[fname] = vy

        for ipage in range(n_pages):
            ore.orbits[fname][ipage] = get_bpms(orb_file, ipage=ipage)

        n_bpms = len(ore.orbits[fname][0][0])
        for i_bpm  in range(n_bpms):
            cx, cy = get_bpm_response(ore.orbits[fname], i_bpm,  ore.scan_values[fname])
            if plane == 'x':
                orm.Rxx[i_cor, i_bpm] = cx
                orm.Rxy[i_cor, i_bpm] = cy
            if plane == 'y':
                orm.Ryx[i_cor, i_bpm] = cx
                orm.Ryy[i_cor, i_bpm] = cy
        i_cor += 1

    return orm, ore


def build_orm(fnames, ref_file_name, plane='x', dkick=1.e-4):
    orb_file = sdds.SDDS(0)
    ref_file  = sdds.SDDS(0)

    ore = OrbitResponse()
    print('opening first measurement file...')
    orb_file.load(fnames[0] + '.clo')
    print('opening reference file...')
    ref_file.load(ref_file_name + '.clo')
    x_, y_ =  get_bpms(orb_file, ipage=0)
    n_bpms = len(x_)

    print('building ORM', len(fnames), n_bpms)

    orm = ORM(len(fnames), n_bpms)


    ore.orbits['ref_orbit'] = get_bpms(ref_file, ipage=0)

    i_cor = 0
    for fname in fnames:
        print('reading', fname + '.clo')
        orb_file.load(fname + '.clo')
        #print(orb_file.columnName)
        ore.hcors.append(fname)
        cname = fname.replace('orm_'+plane+'_','')
        #vx, vy = get_scan_values(cname, param_file, n_pages)
        #if plane == 'x' : ore.scan_values[fname] = vx
        #if plane == 'y' : ore.scan_values[fname] = vy

        ipage = 0
        ore.orbits[fname] = get_bpms(orb_file, ipage=ipage)

        n_bpms = len(ore.orbits[fname][0])
        for i_bpm  in range(n_bpms):
            cx, cy = get_bpm_response(ore.orbits[fname], ore.orbits['ref_orbit'], i_bpm, dkick = dkick)
            if plane == 'x':
                orm.Rxx[i_cor, i_bpm] = cx
                orm.Rxy[i_cor, i_bpm] = cy
            if plane == 'y':
                orm.Ryx[i_cor, i_bpm] = cx
                orm.Ryy[i_cor, i_bpm] = cy
        i_cor += 1

    return orm, ore


def get_orm(fname):
    fi = sdds.SDDS(0)
    fi.load(fname)

    n_cor = len(fi.columnName) - 2
    ipage=0
    n_bpm = len(fi.columnData[0][ipage])
    R = np.zeros([n_cor,n_bpm])
    for i in range(n_cor):
        for j in range(n_bpm):
            R[i,j] = fi.columnData[i+2][ipage][j]
    return R


def get_theor_orm(idx=0,fname='twiss'):
     x = sdds.SDDS(0)
     x.load(fname + '.twi')
     #print(x.columnName)
     #print(x.columnName[0], x.columnName[1], x.columnName[6], x.columnName[7])
     #print(x.columnData[0])
     s = x.columnData[0][idx]
     beta_x = np.array(x.columnData[1][idx])
     beta_y = np.array(x.columnData[7][idx])
     mu_x = np.array(x.columnData[3][idx])
     eta_x = np.array(x.columnData[4][idx])
     mu_y = np.array(x.columnData[9][idx])
     names = np.array(x.columnData[14][idx])
     etypes = np.array(x.columnData[16][idx])

     #print('tunes:', mu_x[-1], mu_y[-1])
     Qx = mu_x[-1] / (2.*pi)
     Qy = mu_y[-1] / (2.*pi)

     L = s[-1]
     #print('L:', L)

     idx = x.parameterName.index('alphac')
     alphac = float(x.parameterData[idx][0])

     #print('alphac:', alphac)


     idx_cor = []
     idx_bpm = []

     for i in range(len(names)):
         if etypes[i] == 'KICKER': idx_cor.append(i)
         if etypes[i] == 'MONI': idx_bpm.append(i)

     Cx = np.zeros([len(idx_cor), len(idx_bpm)])
     Cy = np.zeros([len(idx_cor), len(idx_bpm)])

     Cxy = np.zeros([len(idx_cor), len(idx_bpm)])
     Cyx = np.zeros([len(idx_cor), len(idx_bpm)])

     for i in range(len(idx_cor)):
         for j in range(len(idx_bpm)):
             ii = idx_cor[i]
             ij = idx_bpm[j]
             Cx[i][j] = np.sqrt( beta_x[ii] * beta_x[ij]) * cos(abs(mu_x[ii] - mu_x[ij]) - pi*Qx) / (2*sin(pi*Qx))
             Cy[i][j] = np.sqrt( beta_y[ii] * beta_y[ij]) * cos(abs(mu_y[ii] - mu_y[ij]) - pi*Qy) / (2*sin(pi*Qy))
             #Cx[i][j] += eta_x[ii] * eta_x[ij] / (alphac * L )
             #Cy[i][j] += eta_x[ii] * eta_x[ij] / (alphac * L )

     return Cx, Cy
